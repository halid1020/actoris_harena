# Where this code came from

The shared pipeline under `actoris_harena/{sync.py,recording,training,analysis,
deploy,web,policies}` was written in `so101_garment` and moved here so a second
robot could use it without forking it. It was **copied**, not grafted with
`git subtree` or `git filter-repo`, and this file is the reason.

Preserving per-line history would have meant rewriting this repository's
history and pulling ~64k lines of unrelated simulation commits into the graph
of every future `git log` here. The `so101_garment` history is intact where it
is, and it remains the record of how each of these modules was measured into
existence — most of the claims in its `CLAUDE.md` cite commits in it. This file
is the pointer back.

| Area | Moved in | From `so101_garment` at |
|---|---|---|
| `sync.py`, `deploy/{chunking,chunk_metrics,chunk_sweep,sweep_journal}.py`, `recording/{camera_controls,device_faults}.py` | stage 3a | `83f56db` (`develop`, after the four feature branches merged) |
| `deploy/{policy_run,policy_wire,policy_client,policy_log,policy_rig}.py` | stage 3b | `0fd77dd` (`harena-migration`) |
| `recording/{depth,drift,usb_topology,fault_report,audio_cue,dataset_check,dataset_edit,dataset_read,collection_settings}.py` + `recording/sounds/` | stage 3c | `6719bec` (`harena-migration`) |
| `web/{roots,util,jobs,lifecycle,projects}.py` | stage 3d | `878d8ee` (`harena-migration`) |
| `recording/{dataset_view,camera_profile}.py` | stage 3e | `0080538` (`harena-migration`) |
| `training/{destinations,matrix,metrics,progress,runs}.py` | stage 3f | `ec49dda` (`harena-migration`) |
| `analysis/*` (12 files), `outputs.py`, `action_layout.py` | stage 3g | `3d04f70` (`harena-migration`) |
| `recording/{cameras,realsense_camera,frames}.py` | stage 3h | `657706b` (`harena-migration`) |
| `recording/config.py` (the recording half of `config_parser.py`) | stage 3i | `b522983` (`harena-migration`) |
| `recording/features.py` (with `RobotSchema`) | stage 3j | `cc2b669` (`harena-migration`) |
| `web/{roots_api,lifecycle_api,projects_api,training_api}.py`, `recording/monitor_wire.py` | stage 4 | `0c544e0` (`harena-migration`) |
| `rigs.py`, `cli.py`, `web/console.py` | stage 4 | new here |
| `recording/episode_recorder.py` (with `ObservationBuilder`/`TimedFrameSource`) | stage 3m | `a532cf8` (`develop`) |
| `policies/*` (9 policies), `tool/port_policies.py`, `tool/measure_tactile_border.py` | stage 5 | `d22de3a` (`develop`) |

To read the history of any file here, find its former path under `src/common/`
in that repository and run `git log --follow` on it there.
