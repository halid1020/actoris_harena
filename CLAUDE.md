# CLAUDE.md — working guide for actoris_harena

Background for Claude Code (and humans). Read this before making changes; keep it
current when structure or workflow changes.

## What this repo is, and the split that runs through it

TWO BODIES OF CODE, in one package, with incompatible dependencies.

**The rig pipeline** — `recording/`, `training/`, `analysis/`, `deploy/`, `web/`,
`policies/`, `rigs.py`, `cli.py`, `sync.py`, `outputs.py`, `action_layout.py`.
Data collection, dataset curation, VLA training, policy deployment, the browser
console and every policy the project trains. Shared by two robots:
`../so101_garment` (dual SO-101) and `../ur3e_raven` (single UR3e).

**The simulated arenas** — `arena/`, `agent/`, `utilities/`, `configuration/`,
`registration/`. A sim-RL benchmark framework: SoftGym cloth, Ravens, dm_control,
and about twenty-five agents. 355 files, ~64k lines, years of history.

THEY CANNOT BE INSTALLED TOGETHER, and that is not a bug to fix. MEASURED:
LeRobot pins `numpy>=2.0,<2.3` and `requires-python>=3.12`; the sim stack pins
`numpy<2.0`, `gym==0.26.2`, `robosuite==1.4.1`. So there are two extras:

    pip install -e ".[sim]"    # the arenas
    pip install -e ".[rig]"    # a robot repo's venv

**`__init__.py` IS LAZY (PEP 562) AND MUST STAY SO.** It used to import Agent,
Arena, TrajectoryDataset and all of `api` eagerly, which transitively pulled
torch, pybullet, dm_control and robosuite — so `import actoris_harena.recording`,
one pure parquet module, was impossible in a robot venv. Adding an eager import
back would break every robot repo, and `test/rig/test_packaging.py` runs a
SUBPROCESS to catch it.

## Working style

- Be surgical: make only the changes the task needs.
- **Linting is scoped**, and deliberately. `.pre-commit-config.yaml`'s `files:`
  covers the rig pipeline and `test/rig/` only. Running black over the 355 sim
  files would be a ~64k-line reformat that destroys `git blame` and makes
  `origin/dataset`, `origin/ros1-noetic` and `origin/ros2-humble` unmergeable,
  for no gain.
- **A rig DECLARES; this package never guesses.** Camera profile, output root,
  config paths, checkout, gripper columns, robot schema — each is set by the rig
  and each REFUSES rather than defaulting when it has not been. The reason is in
  `action_layout.py`: a stale gripper-column default would make `perturb`
  occlude a joint and report a share nobody could tell was wrong.
- **No subpackage may be named `data`.** `.gitignore` carries a bare `data`
  pattern, which silently untracked an entire 1 386-line directory once already.

## Environment

- No venv of its own, on purpose: this installs into each robot repo's venv,
  because a rig's venv is what pins its LeRobot checkout and its hardware
  libraries. `make test-unit` finds a sibling rig's venv, takes `PY=<path>`, or
  refuses with the two commands that fix it.
- `make test-unit` / `make test-integration` / `make lint`.
- **The integration tier's checkpoint test SKIPS without a rig's outputs**, and a
  skip there is not a pass — it is the test that makes "a checkpoint trained by
  either implementation loads into the other" checkable:

      RIG_OUTPUT_DIR=~/Projects/so101_garment/outputs make test-integration

## Layout

- `rigs.py` + `cli.py` + `web/console.py` — `actoris-harena console`, which
  serves every robot on the machine and imports NO hardware code. A rig is a
  directory with a `rig.yaml`; anything that touches a device runs as a
  subprocess started with THAT rig's interpreter. One trap already hit:
  `venv/bin/python` is a SYMLINK, so `Path.resolve()` turns it into
  `/usr/bin/python3.12` and hands the agent an environment with none of its rig's
  packages. Paths are made absolute with `normpath` instead, and a test pins it.
- `recording/frames.py` — three contracts, and the distinctions matter.
  `FramePublisher` is what a capture thread writes into; `TimedFrameSource` is
  what a RECORDER reads, by time, so every stream in a frame is sampled at one
  instant rather than being a mix of latest values; `ObservationBuilder` is the
  three questions that are about the robot (state and action, EE, armed).
  `FrameStore` and `TimedFrameStore` implement the first two.
- `recording/features.py` — `RobotSchema`, the layout rule with the width taken
  out of it. `gripper_columns` is DERIVED: `(5, 11)` for two five-joint arms,
  `(6,)` for one six-joint arm.
- `recording/camera_profile.py`, `outputs.py`, `action_layout.py`,
  `recording/config.py`, `training/destinations.py` — the five things a rig
  declares. Each refuses until it is told.
- `provenance.md` in `documents/` — where each area came from, and why these are
  copies rather than a `git subtree` graft.

- `recording/dataset_check.py` — is a dataset whole? The counted
  episodes against the ones in `meta/episodes/`, `data/` and `extra/`, the
  offset invariant, and the repair for an episode nobody wrote. Pure parquet +
  JSON, no LeRobot import, so it can describe a dataset LeRobot refuses to open.
- `recording/dataset_view.py` — camera-ablation **views**: a dataset
  directory naming only some cameras, with the video files symlinked from the
  source (a few MB, not a copy). `meta/info.json` drives
  `LeRobotDatasetMetadata.video_keys`, so an unnamed camera is never decoded AND
  never reaches the policy — no `--policy.input_features` override to keep in
  step. Also collapses a dataset's task strings onto the one covering the most
  frames (position is no guide: on `cube-pnp-new` the typo is registered first).
  Two policies here take a FIXED number of views, and this rig has five cameras,
  so both get an answer here. **pi0.5** has three slots: a camera whose viewpoint
  it knows takes that slot by name (`PI05_SLOTS`), one it has never seen — every
  tactile camera — takes the next free slot in `PI05_SLOT_ORDER`, more cameras
  than slots is refused, and the `slots` column of `runs.tsv` pins it outright.
  **FastWAM** concatenates its cameras into one frame, so `COMPOSITES` tiles the
  four fingertip cameras 2x2 into ONE 224x224 feature (`tactile_quad`); `all`
  never includes a composite, and a composite is the one thing a view cannot
  symlink — it decodes its parts in lockstep and encodes one video with PyAV
  (not the ffmpeg CLI: a compute node has neither). Built by
  `tool/make_camera_view.py`; the cluster job builds one per `cameras` row.
- `training/` — where a training run may be sent, whether it can
  work there, and how it is going. `destinations.py` (pure:
  `src/conf/train_destinations.yaml` validated, the ssh/rsync argv, and what an
  unreachable machine should be told — its paths reach the REMOTE shell
  unquoted so `~`/`$USER` mean the remote home and user, which is why what may
  appear in them is checked at load; `kind: local` is the machine the console
  is on, and `ssh_argv` returning `bash -lc` is the ONE place that kind is
  consulted, so staging, the manifest, the dispatch, the status and the stop
  all work on it unchanged) + `matrix.py` (the `runs.tsv` row model in Python,
  and `row_refusals`, the ONE place a run is judged: unknown policy, a camera
  the dataset lacks, more cameras than the policy has slots, a batch over a
  MEASURED ceiling, a policy this LeRobot has never heard of, and a run that
  would fall back to the CPU) + `progress.py`/`runs.py` (below). Two front
  ends: `tool/train_launch.py` and the console's Training tab, so a run started
  in the browser is the same run. A `-` in the steps/batch column means
  "whatever fits here" and takes the destination's measured ceiling; an
  explicit number is refused if it is over.
- `training/progress.py` + `runs.py` — how far a run has got. The
  driver's LOG is the only metric record that exists (`--wandb.enable=false` is
  unconditional, this LeRobot ships no `SummaryWriter`, and
  `MetricsTracker.to_dict()` returns exactly the right numbers and is never
  called). Two measured facts shape the parser: **`step:` is abbreviated**
  (`format_big_number` prints 10 500 and 10 600 alike as `10K`, so it is a
  label and cannot be a curve's x-axis), and **tqdm is disabled inside Slurm**
  (a thanos/local log is a `\r`-blob whose frames carry the exact step; a
  CREATE log has no exact step at all). So the step is the preceding tqdm
  frame's where there is one, else the line's ordinal x `log_freq` — exact,
  because lerobot logs at `step % log_freq == 0` and nowhere else, and both
  `log_freq` and the total are in the config dump at the head of the log.
  `runs.py` filters the log ON THE FAR SIDE (5.8 MB of frames -> 190 KB) in one
  sentinel-delimited command, discovers run directories rather than listing what
  was launched, and returns **stdout only** — CREATE's stderr is an MFA banner.
  Staleness is judged from the file's mtime against the REMOTE clock, never the
  timestamps inside (they carry no timezone). `runs.py` knows BOTH trees —
  `vla_real_long` and `vla_sim_long`, whose cells are `<mode>/<task>/<policy>`
  rather than `train/<policy>` — and pairs a sim log with its cell by listing
  the DIRECTORY and rebuilding the log's name from it, since every one of the
  three parts may itself contain an underscore (`handover_split`, `harena_act`).
  The same round trip also brings back the resolved `train_config.json` (the
  runs table diffs it) and, for a sim cell, the per-checkpoint rollout results.
- `training/metrics.py` — how a NEW curve gets drawn without editing
  anything. A number crosses three hops before it can be plotted — the grep
  that runs on the far machine, the parser, and the panel grid — and each was a
  closed list. Now a policy prints `harena-metric step=12000 eval/loss=0.0421` (and `so101-metric`, the
  name it used before the pipeline was shared, is still parsed for ever)
  and the curve appears: `KEEP_PATTERN` keeps the sentinel, `progress.py` folds
  it into `series` beside lerobot's own tracker fields, and the page builds its
  panels from `metrics` rather than from names of its own. The namespace
  (`train/`, `eval/`, `pred/`) picks the panel block and is REQUIRED, refused
  where it is emitted rather than guessed at the far end. lerobot's held-out
  validation loss rides the same channel (`step N: eval_loss=…`, whose step is
  EXACT, unlike `step:`); it needs `--eval-split`/`--eval-steps` on the driver
  and is off by default because a validation split holds out episodes and so
  changes what is trained. **Per-checkpoint rollout success is sim-only** —
  `long_vla_sim.sh` writes `val/step_<N>.json`; the real rig's equivalent is a
  person judging trials in `outputs/policy_runs/`, and drawing that on a
  training axis would report a measurement nobody made.
- `analysis/` — **what each input stream contributes to the actions a
  policy plans**, kept OUT of the inference path (nothing there imports it back)
  and driven by `tool/analyse_policy_inputs.py`. It rests on one verified fact:
  both policies reduce their inputs to a vector in which each stream owns a
  CONTIGUOUS piece — ACT's encoder tokens are `[latent, state, cam x H*W, ...]`
  in `config.image_features` order (300 tokens a camera at 480x640, 1 502 in
  total, MEASURED against the loaded backbone), diffusion concatenates
  per-camera features once per observation step — so `streams.py` is that map,
  pure and GPU-free, and `check_layout` asserts it tiles. Four methods, because
  each answers something the others cannot: `perturb.py` (occlusion — behaviour,
  and the ground truth the rest are SCORED against; leave-one-out measures
  redundancy, only-one-in measures sufficiency; the baseline is part of the
  result and every figure names it), `gradients.py` (integrated gradients,
  whose completeness axiom makes per-stream shares parts of one whole;
  SmoothGrad; Grad-CAM), `attention.py` (ACT only, per action of the chunk) and
  `diffusion`'s differences. MEASURED over 206 frames of six episodes on the
  finished ACT checkpoint: **central 57.1 %, proprioception 36.4 %, the four
  fingertips 6.5 % together** — but tactile is not flat, running 0.6 % while the
  arms travel and peaking 21–31 % in every episode, so a mean over an episode
  hides the whole point. During `opening` proprioception rises to 74.7 %. IG
  agrees with occlusion at **+0.90**. ACT's cross-attention has almost no
  DYNAMIC RANGE (pooled shares 0.192–0.219, a 1.14x spread, against occlusion's
  42x) and per FRAME agrees at only +0.17 with 35 % of frames negative — so the
  raw mass is dominated by token count and only the deviation from uniform is
  reported. IG's completeness error at 64 steps averages 0.20 over real frames
  (0.87 worst) though the shares converge by then. `--sanity` is Adebayo et
  al.'s model-randomisation test, which this checkpoint passes outright (a
  randomised policy plans the same chunk whatever it is shown, so every share
  falls to zero). `slides.py` + `tool/analysis_slides.py` turn a finished run
  into slide-ready videos (PyAV/H.264, no ffmpeg binary), plots and tables.
  `paths.py` is the ONE place an analysis decides where it is written
  (`outputs/analysis/<day>/<policy>-<cameras>/`). Not every method exists for
  every architecture and the reasons are structural, so `cam_trunks` is the one
  place Grad-CAM's trunk is found — ACT calls one `model.backbone` per camera,
  diffusion's `rgb_encoder` is an `nn.ModuleList` whose members are called and
  whose `.backbone` is what has a spatial map, and a token model has no map at
  all. pi0.5 carries `@torch.no_grad()` TWICE (on `predict_action_chunk` and
  again on the inner `sample_actions`), so a gradient needs both off. Occlusion
  is pinned through `perturb.plan_of`: it was not, for a while, which made it
  the one method measuring the sampler while being the ground truth everything
  else is scored against. Runbook: `documents/policy_input_analysis.md`.
- `policies/` — every policy this rig trains, implemented HERE rather
  than in LeRobot. Importing the package registers each one with LeRobot's
  draccus registry, which is the whole mechanism: `lerobot.configs.parser.wrap`
  loads whatever `--policy.discover_packages_path` names BEFORE draccus parses,
  and `get_policy_class` then resolves our names by the same route as its own
  (`policies/factory.py:606`). So one policy defined here is trainable by
  `lerobot-train`, servable by `policy_server.py`, fetchable by
  `fetch_policies.sh` and analysable by `analysis/` with no change to any
  of them. The naming is a CONTRACT, not a style — LeRobot derives the policy
  class and the processor factory from the config class name, mechanically:
  `<x>/configuration_<x>.py` holds `Harena<X>Config` registered as `harena_<x>`,
  `modeling_<x>.py` holds `Harena<X>Policy`, `processor_<x>.py` holds
  `make_harena_<x>_pre_post_processors`. `act`, `diffusion`, `pi05` and `fastwam`
  are **ports**: the upstream module tree MOVED, not rewritten, so `state_dict` keys
  are identical and the finished 80 000-step ACT checkpoint loads into either
  implementation (VERIFIED bitwise, not in principle —
  `test/integration/test_policy_ports_checkpoints.py`). `_port.py` holds every
  rule the port applies and `tool/port_policies.py --check` re-derives them, so
  a LeRobot bump is one command and a hand-edit fails
  `test/rig/test_policy_ports.py` immediately. The three ported directories are
  excluded from black/isort/flake8/mypy in `.pre-commit-config.yaml` for that
  reason — reformatting them would destroy the diff against upstream that makes
  the claim checkable; `mypy.ini` skips them at the IMPORT boundary too, because
  that exclude covers the file list and not the import graph, so a subclass
  importing a port drags upstream's type errors in under our name.
  **`fastwam` is the odd port**: it landed upstream AFTER `LEROBOT_COMMIT`, so
  `_port.PORT_REF` names a commit to `git show` that one directory out of rather
  than moving the pin (which would change the other three underneath every
  finished checkpoint). It also carries a `wan/` subpackage verbatim
  (`PORT_EXTRA`, still in `--check`) but NOT upstream's `__init__.py`, which
  re-exports a class the port renames. The pin lacks exactly two names it needs
  — `make_default_policy_processor_steps`, `make_policy_processor_pipelines` —
  reproduced in `policies/common/processor_compat.py` with the ONE import rewritten,
  because editing the file would stop it being a port; a test asserts the pin
  still lacks them so the shim cannot outlive its reason. Before launching one:
  FastWAM pulls a 5B Wan video backbone plus a umt5-xxl text encoder, far larger
  than anything else in the matrix, and may not fit the 24.5 GiB box at any
  batch. `loading.py` registers the package for any loader
  (`eval_sim_policy.load_policy` calls it), and `tool/retarget_checkpoint.py`
  reads a checkpoint through the other member of a ported pair by symlinking its
  weights and rewriting one field — which is how a repo-local pi0.5 gets a base
  to finetune, since `lerobot/pi05_base` says `pi05` and would otherwise quietly
  load LeRobot's class. Three more — `act_crop`, `diffusion_crop`, `pi05_crop` —
  are two-field SUBCLASSES of the ports, adding a registered preprocessor step
  (`policies/common/tactile.py`) that crops the four fingertip cameras to the gel centre
  and RESIZES BACK, so no shape changes anywhere: ACT keeps 300 tokens a camera
  (which `analysis/streams.py` requires), diffusion's dummy-sized feature dim
  still matches, and pi0.5's letterbox padding is unchanged. It goes at index 0,
  ahead of the rename step, or pi0.5's slot rename hides the cameras from it.
  Grad-CAM showing the policy on the sensor EDGES before contact is why. The
  fraction comes from `tool/measure_tactile_border.py`, and measuring it CHANGED
  the design: the rim is a smooth vignette (+7 to +22 % brighter at the edge)
  with no band to find, and HORIZONTALLY the bright rim and the responsive
  columns are the same pixels — on two of four sensors the top-quartile-variation
  columns run to the frame edge. So the default crops ROWS ONLY, `(0.80, 1.00)`,
  inside the 0.62 bound of the tightest camera. `(1.0, 1.0)` is a bit-identical
  uncropped control.
  `flowmatch` is NOT a port: pi0.5's objective and action
  expert on the diffusion policy's `DiffusionRgbEncoder` (that literal class, so
  "same backbone" is a fact), built as the CONTROL for the world action model --
  it shares DreamZero's loss and shares nothing else. **The two flow-matching
  time conventions run OPPOSITE ways** and a model trained in one and sampled in
  the other trains perfectly and emits noise: pi0.5 puts noise at t=1 and
  integrates DOWN, DreamZero's Eq. 2 puts the clean sample at t=1 and integrates
  UP. `policies/common/flow.py` implements pi0.5's and says so; MEASURED 9.1 % of target
  scale the right way against 359.9 % the wrong way. `dreamzero` is the WORLD
  ACTION MODEL (arXiv 2602.15922) at rig scale: video latents and actions
  denoised together under one flow-matching objective, chunk-wise teacher
  forcing, a between-chunk causal mask (`masking.py`, Fig. 14 — printable and
  tested, because the first draft LEAKED a chunk's own clean twin, which is the
  answer it predicts), KV-cache rollout, Flash's decoupled schedules and
  Savitzky-Golay smoothing. It trains from scratch with NO video pretraining, on
  a per-frame VAE rather than Wan's temporal one, at ~1/100th the parameters —
  so it tests the paper's CLAIMS, not its numbers. A chunk's frames and actions
  must span the same interval; `frame_stride` subsamples to make that true and a
  `chunk_size` that will not divide is refused. The frozen VAE is fetched from
  the Hub and kept OUT of the checkpoint. See `documents/policy_package.md` and
  `documents/world_action_model.md`.

## Conventions

- **Pre-commit runs black, isort, flake8, mypy**, scoped as above. Match them.
- **Git: every change set starts on a fresh branch off `develop`.** Finish with a
  commit ending in the `Co-Authored-By: Claude …` trailer. The user pushes.
- The sim side has known latent bugs that are NOT this pipeline's to fix:
  `api.py:43,46` reference an undefined `AGENT_NEEDS_CONFIG`, and seventeen sites
  read `os.environ['actoris_harena_PATH']` (lowercase) against an uppercase
  setter. Recorded so nobody rediscovers them as new.
