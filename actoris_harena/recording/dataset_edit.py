"""Curation helpers for a collected LeRobot v3.0 dataset (edit/delete on disk).

Deleting a recorded episode is not a file removal: v3.0 packs several episodes
into shared chunk files, so a real per-episode delete goes through LeRobot's
``delete_episodes`` (it re-encodes any video segment that mixes kept and deleted
episodes and renumbers the survivors 0..M-1). Our own side files under
``extra/`` (per-episode depth directories, sidecar and drift parquet) and the
two dataset-level meta JSONs LeRobot does not know about are re-indexed here with
the same old->new mapping, then swapped in atomically.

Because that rewrite touches the whole dataset, it costs tens of seconds even
for a single episode, which is far too slow to sit behind a click while an
operator reviews a session. Deletion is therefore two steps. Marking is
instant: the reviewer's decision goes into a small marker file
(``extra/soft_deleted.json``) and the marked episodes are hidden from the review
tool immediately. Compaction performs the real rewrite once for the whole batch.

The marked episodes remain on disk until compaction, so anything that CONSUMES a
dataset must treat a non-empty marker as "not ready": training on a dataset with
pending marks would train on the very episodes the reviewer threw away. Use
``read_soft_deleted`` to check.

The pure helpers (``deletion_mapping``, ``extra_reindex_ops``,
``surviving_indices``) are unit-tested; ``delete_episodes_in_place`` and
``compact_dataset`` perform the file operations. Shared by the web review tool
and any other curation entry point so the re-indexing lives once.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

# Marker file listing episodes the reviewer has deleted but that are still on
# disk. Lives inside the dataset so it travels with it (and so a reviewer cannot
# lose the decision by restarting the tool).
SOFT_DELETE_REL = "extra/soft_deleted.json"


class ReadOnlyDatasetError(OSError):
    """Raised when a delete is attempted on a read-only filesystem.

    Deletion rewrites the dataset in place (a sibling temp dir plus atomic
    renames in the parent directory), so it needs write access to the dataset's
    parent. On a read-only mount that is impossible; callers surface this as a
    clean message instead of a mid-operation ``OSError``.
    """


def writability_problem(path: Path) -> str:
    """Why ``path`` cannot be written to, in the operator's terms, or "".

    "Cannot write here" has two quite different causes on this rig and they need
    opposite remedies. A read-only mount is the drive's fault and is fixed by
    remounting it. Files owned by another user are the MOUNT OPTIONS' fault: a
    filesystem without real ownership, which is what an external NTFS drive is,
    invents an owner from the uid the mount was given, so a drive mounted by root
    without a uid option presents every file as root's and refuses the person
    sitting at the machine. Telling them to remount read-write, as this used to,
    is advice for a problem they do not have.
    """
    path = Path(path)
    if not path.exists():
        return f"{path} does not exist"
    if os.access(path, os.W_OK):
        return ""
    if os.statvfs(path).f_flag & os.ST_RDONLY:
        return (
            f"{path} is on a read-only filesystem — remount it read-write to "
            "curate this dataset"
        )
    try:
        owner = path.stat().st_uid
    except OSError:
        return f"{path} is not writable"
    if owner != os.getuid():
        mountpoint, fstype = mount_of(path)
        if fstype in _INVENTED_OWNERSHIP:
            return (
                f"{path} is owned by uid {owner}, not you (uid {os.getuid()}) — "
                f"the drive ({fstype}) is mounted without your ownership. Remount "
                f"it with your uid, e.g. sudo mount -o remount,uid={os.getuid()},"
                f"gid={os.getgid()} {mountpoint}"
            )
        # A filesystem with real ownership: the directory was made by someone
        # else, most often `sudo mkdir`. No mount option changes that.
        return (
            f"{path} is owned by uid {owner}, not you (uid {os.getuid()}) — give "
            f"it to yourself: sudo chown -R {os.getuid()}:{os.getgid()} {path}"
        )
    return f"{path} is not writable by you"


#: Filesystems with no ownership of their own, which invent an owner from the
#: uid the mount was given. Anywhere else, a file's owner is real.
_INVENTED_OWNERSHIP = frozenset({"vfat", "exfat", "ntfs", "ntfs3", "fuseblk", "msdos"})


def mount_of(path: Path) -> "tuple[str, str]":
    """``(mountpoint, fstype)`` of the filesystem holding ``path``.

    The longest mountpoint in /proc/mounts that contains it; ``("", "")`` off
    Linux, where the advice falls back to the real-ownership case.
    """
    best = ("", "")
    try:
        lines = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return best
    target = os.path.realpath(path)
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mountpoint = fields[1].replace("\\040", " ")
        inside = target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/")
        if inside and len(mountpoint) > len(best[0]):
            best = (mountpoint, fields[2])
    return best


def deletion_mapping(total: int, to_delete: "list[int]") -> "dict[int, int]":
    """``{old_index: new_index}`` for the episodes KEPT after a deletion.

    Kept episodes stay in ascending order and are renumbered 0..M-1, exactly
    matching LeRobot's own re-indexing in ``delete_episodes`` so our side files
    line up with the rewritten dataset. Deleted indices are absent from the map.
    Pure.
    """
    dead = set(to_delete)
    kept = [i for i in range(total) if i not in dead]
    return {old: new for new, old in enumerate(kept)}


def episode_uid_rel(index: int) -> str:
    """Path (relative to the dataset root) of one episode's identity file. Pure."""
    return f"extra/uid_{index:06d}.json"


def new_episode_uid(when: float | None = None) -> str:
    """A unique, sortable id for a recording, from the wall clock it started at.

    ``YYYYmmdd-HHMMSS-mmm``. The episode index cannot serve as identity because
    deleting an episode renumbers every later one, so the same index names a
    different recording afterwards; this id never moves. Millisecond precision
    keeps it unique even if two episodes start within the same second.
    """
    stamp = time.time() if when is None else when
    millis = int((stamp % 1.0) * 1000.0)
    return f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(stamp))}-{millis:03d}"


def write_episode_uid(root: Path, index: int, uid: str, task: str = "") -> Path:
    """Record ``uid`` as episode ``index``'s identity. Returns the file written."""
    path = Path(root) / episode_uid_rel(index)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_uid": uid,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task,
    }
    tmp = path.with_suffix(".json.partial")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def read_episode_uid(root: Path, index: int) -> str:
    """Episode ``index``'s recorded id, or "" when it has none. Never raises."""
    try:
        with open(Path(root) / episode_uid_rel(index), "r") as f:
            return str(json.load(f).get("episode_uid", ""))
    except (OSError, ValueError, TypeError, AttributeError):
        return ""


def extra_reindex_ops(
    mapping: "dict[int, int]", depth_names: "list[str]"
) -> "list[tuple[str, str]]":
    """``(src_rel, dst_rel)`` moves that re-index our ``extra/`` side files.

    For every kept episode ``old -> new`` this maps the per-episode drift and
    sidecar parquet, its identity file, and each depth stream's per-episode
    directory from its old six-digit index to its new one. Paths are relative to
    the dataset root so a caller joins them with the source and destination
    roots; missing sources are skipped by the caller. Pure.
    """
    ops: list[tuple[str, str]] = []
    for old, new in sorted(mapping.items()):
        ops.append((f"extra/drift_{old:06d}.parquet", f"extra/drift_{new:06d}.parquet"))
        ops.append(
            (f"extra/episode_{old:06d}.parquet", f"extra/episode_{new:06d}.parquet")
        )
        ops.append((episode_uid_rel(old), episode_uid_rel(new)))
        for dname in depth_names:
            ops.append(
                (
                    f"extra/depth/{dname}/episode_{old:06d}",
                    f"extra/depth/{dname}/episode_{new:06d}",
                )
            )
    return ops


def surviving_indices(total: int, marked: "list[int]") -> "list[int]":
    """Episode indices still visible after ``marked`` are hidden. Pure.

    Unlike ``deletion_mapping`` this does NOT renumber: a soft delete only hides
    episodes, so the survivors keep the on-disk indices that the video and
    per-episode routes need. Renumbering happens once, at compaction.
    """
    dead = set(marked)
    return [i for i in range(total) if i not in dead]


def read_soft_deleted(root: Path) -> "list[int]":
    """Episodes marked for deletion but still on disk. Never raises.

    A missing, empty or corrupt marker reads as "nothing marked": losing a
    reviewer's marks is far better than refusing to open the dataset, and the
    episodes themselves are still intact either way.
    """
    path = Path(root) / SOFT_DELETE_REL
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return sorted({int(i) for i in data.get("episodes", [])})
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def write_soft_deleted(root: Path, indices: "list[int]") -> "list[int]":
    """Replace the marker with ``indices`` (sorted, de-duplicated). Returns it.

    Writes via a temp file and an atomic rename inside ``extra/`` so an
    interrupted write cannot leave a half-written marker. Needs only the dataset
    to be writable, not its parent, so marking works on mounts where the
    in-place rewrite of ``delete_episodes_in_place`` could not run.
    """
    root = Path(root)
    clean = sorted({int(i) for i in indices})
    path = root / SOFT_DELETE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.partial")
    payload = {"episodes": clean, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return clean


def clear_soft_deleted(root: Path) -> None:
    """Remove the marker file, if present."""
    path = Path(root) / SOFT_DELETE_REL
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def compact_dataset(root: Path, repo_id: str, depth_names: "list[str]") -> int:
    """Really delete every marked episode and clear the marker. Returns the new N.

    This is the expensive half of the two-step delete: it re-encodes the video
    segments that mixed kept and deleted episodes and renumbers the survivors, so
    it is run once for a whole batch of marks rather than per episode.
    """
    marked = read_soft_deleted(root)
    if not marked:
        return saved_episode_total(root)
    remaining = delete_episodes_in_place(root, repo_id, marked, depth_names)
    # The rewrite produced a fresh dataset directory, so the marker is already
    # gone with the old one; clear defensively in case a future rewrite copies
    # extra/ wholesale.
    clear_soft_deleted(root)
    return remaining


def saved_episode_total(root: Path) -> int:
    """Episode count recorded in the dataset's own metadata (0 if unreadable)."""
    try:
        with open(Path(root) / "meta" / "info.json", "r") as f:
            return int(json.load(f).get("total_episodes", 0))
    except (OSError, ValueError, TypeError):
        return 0


def episode_lengths(meta: Any) -> "list[int]":
    """Per-episode frame counts from a ``LeRobotDatasetMetadata``. Pure-ish."""
    return [int(meta.episodes[k]["length"]) for k in range(meta.total_episodes)]


def commit_episode_metadata(dataset: Any) -> bool:
    """Make the episode LeRobot just saved durable on disk. Returns whether it did.

    LeRobot keeps a recorded episode in RAM longer than it looks. The metadata
    side buffers ten episodes before writing anything and holds its parquet
    writer open until the dataset is finalized at exit; the frame-data side keeps
    its own writer open until one file reaches a size limit. So while a session
    runs, ``meta/episodes/`` is empty or footerless and the newest
    ``data/`` file is footerless too -- which is all that a "Parquet magic bytes
    not found in footer" complaint means. A reader cannot list the episodes
    (hence frame counts showing as unknown) and cannot open the dataset at all,
    because reading concatenates every data file and one unfinished file fails
    the lot. Worse, an interrupted collector -- a kill, a power cut, a crash past
    the exit handler -- loses whatever never reached disk, and an episode whose
    metadata or frames never landed is unusable for training even though its
    video is sitting on the drive.

    Each saved episode is therefore committed here, on both sides: close the
    writers so the footers are written, then point the NEXT episode at a fresh
    file. Advancing is required rather than tidy -- LeRobot reopens a writer at
    the path derived from the last episode, which would truncate the file just
    closed and take the earlier episodes with it. For the frame data that means
    handing control back to LeRobot's own "start a new file" branch (the one that
    runs when a dataset is resumed) by clearing its cached last episode and
    seeding ``meta.episodes`` with the row just committed, so the new file's
    indices and frame offsets are computed by LeRobot rather than by us. One
    modest file per episode is exactly the layout resuming already produces, and
    ``read_episode_lengths`` reads such a directory with per-file fault
    isolation.

    Returns ``False`` when the object does not expose the internals this relies
    on (a different LeRobot version, or a stub in tests), so a caller can carry on
    with exit-time finalization as the fallback rather than fail an episode.
    """
    meta: Any = getattr(dataset, "meta", None)
    close_meta_writer = getattr(meta, "_close_writer", None)
    latest = getattr(meta, "latest_episode", None)
    if meta is None or close_meta_writer is None or not isinstance(latest, dict):
        return False
    if not {"meta/episodes/chunk_index", "meta/episodes/file_index"} <= set(latest):
        return False
    close_meta_writer()
    committed = _last_committed_episode(getattr(meta, "root", None), latest)
    # The row already written keeps the indices it was written with; only the
    # in-memory "where the next one goes" pointer moves.
    chunk_index = int(latest["meta/episodes/chunk_index"][0])
    file_index = int(latest["meta/episodes/file_index"][0])
    chunks_size = int(getattr(meta, "chunks_size", 0) or 1000)
    if file_index >= chunks_size - 1:
        chunk_index, file_index = chunk_index + 1, 0
    else:
        file_index += 1
    latest["meta/episodes/chunk_index"] = [chunk_index]
    latest["meta/episodes/file_index"] = [file_index]

    # Frame data: only safe once the metadata row above is on disk, since that
    # row is what LeRobot reads to place the next data file.
    writer: Any = getattr(dataset, "writer", None)
    close_data_writer = getattr(writer, "close_writer", None)
    if close_data_writer is not None and committed is not None:
        close_data_writer()
        meta.episodes = [committed]
        writer._latest_episode = None
    return True


def _last_committed_episode(
    root: Any, latest: "dict[str, Any]"
) -> "dict[str, Any] | None":
    """The episode row just written, read back from its own parquet file.

    LeRobot's "start a new data file" branch reads the previous episode's frame
    offsets and file indices from ``meta.episodes[-1]``. Only the last row is
    needed, and it lives alone in the file just closed, so this is a single small
    read rather than the whole episode history. Returns ``None`` if the row
    cannot be read, which keeps the caller from touching the data writer at all.
    """
    if root is None:
        return None
    rel = "meta/episodes/chunk-{:03d}/file-{:03d}.parquet".format(
        int(latest["meta/episodes/chunk_index"][0]),
        int(latest["meta/episodes/file_index"][0]),
    )
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(Path(root) / rel).to_pylist()
    except Exception:
        return None
    return rows[-1] if rows else None


def read_episode_lengths(root: Path) -> "tuple[dict[int, int], list[str]]":
    """``({episode_index: length}, [unreadable files])`` straight from the parquet.

    Building a ``LeRobotDatasetMetadata`` just to list episodes is expensive (it
    loads the episode metadata through HuggingFace ``datasets``: measured at
    ~4.9 s for 58 episodes, against ~0.3 s here) and it is all-or-nothing -- one
    truncated parquet raises and no episode can be listed at all. Reading the
    columns directly is both quicker and per-file fault tolerant, which is what a
    review tool needs: a half-written file from an interrupted session should cost
    the episodes in THAT file, not the whole session.

    Returns the lengths it could read and the files it could not, so the caller
    can tell the operator which part of the dataset is damaged.
    """
    root = Path(root)
    lengths: dict[int, int] = {}
    bad: list[str] = []
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.is_dir():
        return lengths, bad
    import pyarrow.parquet as pq  # local: keeps the import off pure-helper users

    for path in sorted(meta_dir.rglob("*.parquet")):
        try:
            table = pq.read_table(path, columns=["episode_index", "length"])
            data = table.to_pydict()
            for index, length in zip(data["episode_index"], data["length"]):
                lengths[int(index)] = int(length)
        except Exception:
            # Any failure to read is the same outcome for the caller: these
            # episodes' lengths are unknown. Keep going with the rest.
            bad.append(str(path.relative_to(root)))
    return lengths, bad


# ── Episodes-metadata repair ─────────────────────────────────────────────────
#
# Every row of ``meta/episodes/<chunk>/<file>.parquet`` carries the chunk and
# file index of the file it is stored in. That pair is self-referential, so a
# row can always be checked -- and corrected -- against its own path.
#
# It needs correcting because LeRobot's ``aggregate_datasets`` merges episode
# metadata by copying each source's rows and merely OFFSETTING that column
# (``datasets/aggregate.py``), while writing every row into the destination's
# first file. A merged dataset therefore claims its episodes live in files that
# were never written, and the next rewrite -- a compaction, or a second merge --
# fails when ``dataset_tools._load_episode_with_stats`` opens one of them.
# MEASURED on a merged dataset: 62 rows in ``file-000.parquet`` claiming file
# indices 0..55. Freshly recorded datasets are consistent, so this repairs
# damage rather than papering over the recorder.

EPISODES_GLOB = "meta/episodes/chunk-*/file-*.parquet"
META_CHUNK_KEY = "meta/episodes/chunk_index"
META_FILE_KEY = "meta/episodes/file_index"
_META_FILE_RE = re.compile(r"chunk-(\d+)/file-(\d+)\.parquet$")


def meta_file_indices(path: Path) -> "tuple[int, int]":
    """The ``(chunk, file)`` a metadata file's own path declares. Pure."""
    match = _META_FILE_RE.search(Path(path).as_posix())
    if match is None:
        raise ValueError(f"not an episodes-metadata file: {path}")
    return int(match.group(1)), int(match.group(2))


def episode_meta_files(root: Path) -> "list[Path]":
    """Every episodes-metadata parquet in the dataset, in path order."""
    return sorted(Path(root).glob(EPISODES_GLOB))


def repair_episode_metadata(root: Path) -> "list[str]":
    """Make every episodes-metadata row name the file it is really in.

    Returns the files it rewrote (empty when the dataset was already
    consistent), so a caller can say what it did. Only the two self-referential
    index columns are touched, and the table is rewritten through pyarrow rather
    than pandas so that every other column -- the per-episode statistics above
    all -- keeps its exact type instead of being round-tripped through object
    arrays. Each file is replaced atomically, so an interrupted repair leaves
    the dataset as it was.
    """
    import pyarrow as pa  # type: ignore[import]
    import pyarrow.parquet as pq  # type: ignore[import]

    fixed: list[str] = []
    root = Path(root)
    for path in episode_meta_files(root):
        chunk, index = meta_file_indices(path)
        table = pq.read_table(path)
        if META_CHUNK_KEY not in table.column_names:
            continue  # not a v3.0 episodes table; leave it alone
        wanted = {META_CHUNK_KEY: chunk, META_FILE_KEY: index}
        if all(
            table.column(key).to_pylist() == [value] * table.num_rows
            for key, value in wanted.items()
        ):
            continue
        for key, value in wanted.items():
            position = table.column_names.index(key)
            field = table.schema.field(position)
            column = pa.array([value] * table.num_rows, type=field.type)
            table = table.set_column(position, field, column)
        temp = path.with_suffix(".parquet.partial")
        pq.write_table(table, temp)
        os.replace(temp, path)
        fixed.append(str(path.relative_to(root)))
    return fixed


def delete_episodes_in_place(
    root: Path, repo_id: str, indices: "list[int]", depth_names: "list[str]"
) -> int:
    """Delete ``indices`` on the drive and renumber the rest. Returns the new N.

    Builds a re-indexed copy in a sibling temp dir (LeRobot's ``delete_episodes``
    for data/videos/meta, plus our ``extra/`` side files and the two extra meta
    JSONs it does not carry over), then atomically swaps it in. The original is
    moved aside first and only removed once the swap succeeds, so a failure
    partway never loses the source dataset. Accepts one or many indices so a
    batch delete is a single re-encode + re-index + swap.
    """
    root = Path(root)
    indices = sorted({int(i) for i in indices})
    if not indices:
        raise ValueError("no episodes given to delete")

    # The rewrite creates a sibling temp dir and renames within the parent, so
    # a read-only mount cannot be edited. Fail early with a clear message rather
    # than partway through LeRobot's re-encode (and before loading the dataset).
    problem = writability_problem(root.parent)
    if problem:
        raise ReadOnlyDatasetError(f"cannot rewrite the dataset: {problem}")

    # Before anything reads the metadata: a dataset produced by a merge may
    # claim its episodes live in metadata files that were never written, and
    # LeRobot would open one of them partway through the rewrite. The dataset
    # object caches what it loads, so the repair has to come first.
    repair_episode_metadata(root)

    # And before LeRobot is constructed at all: an episode this dataset counts
    # but never wrote makes LeRobot judge the whole local copy incomplete and
    # reach for the Hub, which offline reports a local gap as an unreachable
    # huggingface.co. Say what is actually wrong instead.
    from actoris_harena.recording.dataset_check import ensure_loadable

    ensure_loadable(root)

    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    total = ds.meta.total_episodes
    if indices[0] < 0 or indices[-1] >= total:
        raise ValueError(f"episode index out of range 0..{total - 1}: {indices}")
    if len(indices) >= total:
        raise ValueError("cannot delete all episodes in a dataset")

    mapping = deletion_mapping(total, indices)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    temp = root.parent / f"{root.name}.tmp-{stamp}"

    # Everything up to the swap builds the temp dataset, and a failure anywhere
    # in it (a re-encode that dies, a full disk) must not leave tens of
    # gigabytes behind under a name nothing will ever look at again.
    try:
        delete_episodes(ds, indices, output_dir=temp, repo_id=repo_id)

        # LeRobot rewrites meta/ from scratch, so our two dataset-level JSONs
        # are not carried over — copy them across.
        for fn in ("realsense.json", "action_space.json"):
            src = root / "meta" / fn
            if src.is_file():
                (temp / "meta").mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, temp / "meta" / fn)

        # Re-index our side files (extra/) into the temp dataset per the mapping.
        for src_rel, dst_rel in extra_reindex_ops(mapping, depth_names):
            src = root / src_rel
            if not src.exists():
                continue
            dst = temp / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    trash = root.parent / f"{root.name}.trash-{stamp}"
    os.replace(root, trash)  # atomic within the filesystem
    os.replace(temp, root)
    shutil.rmtree(trash, ignore_errors=True)
    return total - len(indices)
