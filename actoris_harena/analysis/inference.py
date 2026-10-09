"""One checkpoint, held open, answering "what would you plan from this?".

The seam every method in this package shares. It is deliberately NOT
``actoris_harena.deploy.policy_client``: that one owns a queue, a session and a chunking
strategy because a rollout needs them, whereas an attribution needs one
observation to give one chunk, repeatably, with nothing remembered in between.

``predict_action_chunk`` is what makes that possible -- it returns the whole
plan and touches no queue -- and for ACT it is deterministic: the VAE latent is
sampled only ``if self.training``, so at eval it is zeros and two identical
observations give identical chunks. Diffusion is not; see
:mod:`actoris_harena.analysis.diffusion`, which pins its starting noise.

Actions come back UNNORMALISED, in the units the arms are commanded in
(degrees, and gripper open fraction), because a per-joint number is meant to be
read. The post-processor unnormalises one step at a time -- handing it a whole
chunk is the obvious thing to try and is wrong -- so a chunk is pushed through
it step by step, exactly as ``tool/policy_server.py`` does.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
from lerobot.utils.constants import OBS_IMAGES

from actoris_harena.analysis.streams import (
    CAMERA_PREFIX,
    act_layout,
    camera_names,
    check_layout,
    diffusion_layout,
    streams_of,
    token_layout,
)


def rename_map_of(pre) -> "dict[str, str]":
    """The observation rename the checkpoint was trained with, if any.

    Read off the saved preprocessor rather than the training config, because the
    preprocessor is what actually runs -- a config that disagreed with it would
    be describing a run that did not happen.
    """
    for step in getattr(pre, "steps", []) or []:
        mapping = getattr(step, "rename_map", None)
        if mapping:
            return dict(mapping)
    return {}


def source_cameras(
    policy_cameras: "list[str]", rename: "dict[str, str]"
) -> "list[str]":
    """Camera names as the DATASET has them, in the policy's own order.

    ``rename`` maps source key -> policy key, so it is inverted here. A camera
    the map does not mention is already named the same on both sides, which is
    every checkpoint that was not finetuned onto pretrained slots.
    """
    inverse = {v: k for k, v in rename.items()}
    out = []
    for name in policy_cameras:
        key = CAMERA_PREFIX + name
        out.append(inverse.get(key, key)[len(CAMERA_PREFIX) :])
    return out


def renamed_streams(streams: list, cameras: "list[str]") -> list:
    """Report each camera under the rig's name, not the policy's slot name.

    A deck saying `base_0_rgb` names something the operator cannot point at; the
    same stream is `central` on the rig. Stream is frozen, so this rebuilds
    rather than mutates. Only the NAME changes -- ``key`` stays the policy-side
    batch key, which is what the conditioning layout is indexed by.
    """
    import dataclasses

    out, i = [], 0
    for stream in streams:
        if stream.kind == "camera" and i < len(cameras):
            out.append(dataclasses.replace(stream, name=cameras[i]))
            i += 1
        else:
            out.append(stream)
    return out


class Inference:
    """A loaded checkpoint and the shapes it implies."""

    def __init__(
        self,
        checkpoint: str,
        device: "str | None" = None,
        task: str = "",
        *,
        load_policy=None,
        build_batch=None,
    ):
        """``load_policy`` and ``build_batch`` come from the caller.

        Loading a checkpoint means knowing which policy package registers its
        type, and assembling a batch means knowing the rig's camera keys and
        state width -- both of which are the caller's, not this package's. They
        used to be imported from ``tool/eval_sim_policy.py``, which made an
        analysis module depend on one repo's tools.
        """
        import torch

        if load_policy is None or build_batch is None:
            raise TypeError(
                "Inference needs load_policy= and build_batch=: the rig's own "
                "checkpoint loader and batch assembler. so101_garment passes "
                "tool/eval_sim_policy.py's pair."
            )

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # from_pretrained treats a relative path as a Hub repo id and fails with
        # a message about repo names, which says nothing about the real cause.
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        if not os.path.isdir(self.checkpoint):
            raise SystemExit(f"❌ no checkpoint directory at {self.checkpoint}")
        self.policy, self.pre, self.post, self.type = load_policy(
            self.checkpoint, self.device
        )
        self.policy.eval()
        self._build_batch = build_batch
        self.task = task
        self.cfg = self.policy.config
        # A pi0.5 finetune was trained with --rename_map, so its config names
        # openpi's SLOTS (base_0_rgb, left_wrist_0_rgb, ...) and not the rig's
        # cameras. Asking the dataset for those fails with "this dataset has no
        # camera base_0_rgb", which reads as a missing camera rather than as a
        # renamed one. The checkpoint carries the map in its own preprocessor,
        # so invert it: fetch under the rig's names, and let `self.pre` rename
        # them exactly as it did in training.
        self.rename = rename_map_of(self.pre)
        self.cameras = source_cameras(camera_names(self.cfg), self.rename)
        self.streams = renamed_streams(streams_of(self.cfg), self.cameras)
        self.action_dim = int(self.cfg.output_features["action"].shape[0])
        self.n_action_steps = int(getattr(self.cfg, "n_action_steps", 1) or 1)
        self.n_obs_steps = int(getattr(self.cfg, "n_obs_steps", 1) or 1)
        self.image_hw = tuple(list(self.cfg.image_features.values())[0].shape[1:])

    # -- what the policy's conditioning looks like -------------------------
    def tokens_per_camera(self) -> int:
        """MEASURED, not predicted: run one frame through the real backbone.

        ``replace_final_stride_with_dilation`` quadruples this and nothing in
        the declared image shape says so, and a token count that is wrong by any
        amount does not raise -- it silently attributes the tail of one camera to
        the head of the next, and the figure still looks plausible.
        """
        backbone = getattr(getattr(self.policy, "model", None), "backbone", None)
        if backbone is None:
            return 0
        height, width = self.image_hw
        with self.torch.no_grad():
            probe = self.torch.zeros(1, 3, height, width, device=self.device)
            feature_map = backbone(probe)["feature_map"]
        return int(feature_map.shape[-2] * feature_map.shape[-1])

    def tokens_per_camera_vlm(self) -> int:
        """A token model's patch count per camera. MEASURED from the tower.

        The declared image shape says nothing about it -- pi0.5 resizes to its
        own resolution before patching, so a 480x640 camera and a 224x224 one
        produce the same number of tokens. Asking the model is the only way to
        be right, and a count that is wrong does not raise: it silently
        attributes the tail of one camera to the head of the next.
        """
        config = self.cfg
        size = int(getattr(config, "resize_imgs_with_padding", None) or 224)
        if isinstance(getattr(config, "resize_imgs_with_padding", None), (tuple, list)):
            size = int(config.resize_imgs_with_padding[0])
        patch = int(getattr(config, "patch_size", 0) or 14)
        side = max(size // patch, 1)
        return side * side

    def camera_feature_dim(self) -> int:
        """Diffusion's per-camera encoder width, from the model rather than a guess."""
        encoder = getattr(getattr(self.policy, "diffusion", None), "rgb_encoder", None)
        if encoder is None:
            return 0
        if isinstance(encoder, (list, tuple)) or hasattr(encoder, "__getitem__"):
            try:
                return int(encoder[0].feature_dim)
            except (TypeError, IndexError, AttributeError):
                pass
        return int(getattr(encoder, "feature_dim", 0))

    def layout(self):
        """Where each stream sits in this policy's conditioning, and whether it tiles.

        Three architectures, named. The fork used to be ``act`` versus
        everything-else-is-diffusion, which handed a pi0.5 checkpoint a layout of
        zero-width camera spans and reported it as a tiling problem rather than
        as the wrong question.
        """
        family = self.family()
        if family == "act":
            per_camera = self.tokens_per_camera()
            spans = act_layout(self.cfg, per_camera)
            total = 1 + sum(s.width for s in spans)
            return self._named(spans), total, check_layout(spans, total, latent=1)
        if family == "diffusion":
            spans = diffusion_layout(self.cfg, self.camera_feature_dim())
            total = sum(s.width for s in spans)
            return self._named(spans), total, check_layout(spans, total)
        spans = token_layout(self.cfg, self.tokens_per_camera_vlm())
        total = sum(s.width for s in spans)
        return self._named(spans), total, check_layout(spans, total)

    def _named(self, spans: list) -> list:
        """Give each span the same stream name everything else here uses.

        The layout functions build their own Streams from the config, so on a
        checkpoint finetuned onto renamed slots they carry `base_0_rgb` while
        every per-frame result is keyed `central`. The deck then died on a
        KeyError naming a camera the operator has never heard of. One naming,
        decided in one place.
        """
        import dataclasses

        by_key = {s.key: s for s in self.streams}
        return [
            dataclasses.replace(span, stream=by_key.get(span.stream.key, span.stream))
            for span in spans
        ]

    def family(self) -> str:
        """Which conditioning layout this checkpoint has: act, diffusion or tokens.

        By STRUCTURE, not by name, so a ported policy (``so101_act``) and a
        renamed one answer the same as the original. The name is the hint; the
        module tree is the evidence.
        """
        policy = self.policy
        if getattr(getattr(policy, "model", None), "backbone", None) is not None:
            return "act"
        if getattr(policy, "diffusion", None) is not None:
            return "diffusion"
        return "tokens"

    def stat_mean(self, key: str) -> "np.ndarray | None":
        """The training mean of ``key`` (e.g. ``observation.state``), raw units.

        Read off the checkpoint's own normaliser, which is the one statistic
        the model was trained against. None when no step carries it.
        """
        for step in getattr(self.pre, "steps", []) or []:
            stats = getattr(step, "stats", None) or {}
            mean = (stats.get(key) or {}).get("mean")
            if mean is not None:
                values = (
                    mean.detach().cpu().numpy() if hasattr(mean, "detach") else mean
                )
                return np.asarray(values, dtype=np.float32)
        return None

    # -- the forward pass ---------------------------------------------------
    def batch(self, state: np.ndarray, images: "dict[str, np.ndarray]") -> dict:
        """One observation -> the batch the policy expects, normalised.

        A policy with more than one observation step was trained on ADJACENT
        frames, so a single observation is repeated to fill the window. That is
        an approximation and it is stated rather than hidden: for a still scene
        it is exact, and for a moving one it removes the velocity cue, which is
        itself worth knowing when a diffusion attribution reads oddly.
        """
        missing = [c for c in self.cameras if c not in images]
        if missing:
            raise ValueError(
                f"this checkpoint needs {', '.join(missing)}, which the "
                f"observation does not have (it has {', '.join(sorted(images))})"
            )
        batch = self._build_batch(state, images, self.task, self.device)
        batch = self.pre(batch)
        # The saved pre-processor carries the device it was TRAINED with -- cuda,
        # for every checkpoint here -- and moves the batch there regardless of
        # what this process asked for. On a machine with a GPU that is a silent
        # half-move: the weights are where we put them and the batch is not, and
        # the forward pass dies deep inside a Linear with a device mismatch that
        # names neither. Put it back where the weights are.
        batch = self.to_device(batch)
        if self.n_obs_steps > 1:
            batch = self._stack_window(batch)
        return batch

    def to_device(self, batch: dict) -> dict:
        """Every tensor in the batch onto this run's device. Lists included."""
        out = {}
        for key, value in batch.items():
            if self.torch.is_tensor(value):
                out[key] = value.to(self.device)
            elif isinstance(value, list) and value and self.torch.is_tensor(value[0]):
                out[key] = [v.to(self.device) for v in value]
            else:
                out[key] = value
        return out

    def _stack_window(self, batch: dict) -> dict:
        """Repeat one observation into the window a multi-step policy expects."""
        out = dict(batch)
        for key, value in batch.items():
            if not self.torch.is_tensor(value):
                continue
            if key.startswith("observation."):
                out[key] = value.unsqueeze(1).repeat_interleave(self.n_obs_steps, dim=1)
        return out

    def chunk(self, state: np.ndarray, images: "dict[str, np.ndarray]") -> np.ndarray:
        """The plan this observation produces: ``(n_action_steps, action_dim)``."""
        return self.chunk_from(self.batch(state, images))

    def chunk_from(self, batch: dict) -> np.ndarray:
        """As :meth:`chunk`, from an already-built batch (so it can be perturbed).

        Note this does NOT pin a stochastic sampler: two calls on one diffusion
        checkpoint differ by roughly the size of the effect an ablation measures.
        Use :func:`actoris_harena.analysis.diffusion.plan` where that matters, which is
        everywhere two plans are compared.
        """
        with self.torch.no_grad():
            planned = self.policy.predict_action_chunk(batch)
        return self.finish_chunk(planned)

    def finish_chunk(self, planned) -> np.ndarray:
        """A raw plan to unnormalised numpy: ``(n_action_steps, action_dim)``.

        Separate so that a seeded forward pass and an ordinary one cannot
        disagree about the trimming or the unnormalising -- the post-processor
        takes one step at a time, and handing it a whole chunk is the obvious
        thing to try and is wrong.
        """
        torch = self.torch
        with torch.no_grad():
            if planned.ndim != 3:
                planned = planned.unsqueeze(0)
            planned = planned[:, : self.n_action_steps, :]
            steps = [self.post(planned[:, i, :]) for i in range(planned.shape[1])]
            out = torch.stack(steps, dim=1).squeeze(0)
        return np.asarray(out.detach().to("cpu"), dtype=np.float32).reshape(
            -1, self.action_dim
        )

    def stochastic(self) -> bool:
        """Does this checkpoint's sampler start from noise? See analysis.diffusion."""
        from actoris_harena.analysis.diffusion import is_stochastic

        return is_stochastic(self)

    # -- the same forward pass, differentiable ------------------------------
    def chunk_tensor(self, batch: dict):
        """The plan as a tensor with gradients attached. Normalised units.

        ``predict_action_chunk`` is decorated ``@torch.no_grad()`` on both
        policies, so a gradient method cannot use it and has to call the model
        underneath by the same route the wrapper does -- including the stacking
        of ``OBS_IMAGES``, which is where the camera ORDER is fixed and so is not
        a detail that may be reproduced loosely.

        The output is left NORMALISED. The post-processor is affine, so
        unnormalising would only rescale every attribution by a constant per
        channel, and doing it per step would put a hundred extra nodes in the
        graph for no change in what is ranked.
        """
        torch = self.torch
        batch = dict(batch)
        keys = list(self.cfg.image_features)
        family = self.family()
        if family == "act":
            batch[OBS_IMAGES] = [batch[k] for k in keys]
            return self.policy.model(batch)[0]
        if family == "diffusion":
            # Diffusion stacks along a camera axis instead of using a list, and
            # its sampler is stochastic -- actoris_harena.analysis.diffusion pins the
            # noise so a gradient is taken through one fixed path, not a lottery.
            batch[OBS_IMAGES] = torch.stack([batch[k] for k in keys], dim=-4)
            from actoris_harena.analysis.diffusion import pinned

            # A gradient has to be taken through ONE fixed sampling path, not a
            # lottery -- the scheduler draws fresh noise at every denoising step.
            with pinned(self):
                return self.policy.diffusion.generate_actions(batch)
        # A token model (pi0.5 and the flow-matching policies here) has no single
        # module to reach past the no_grad wrapper for, so the wrapper is
        # bypassed directly. That is legitimate -- the decorator is there to save
        # memory in a rollout, not to mark the pass as non-differentiable.
        inner = getattr(self.policy.predict_action_chunk, "__wrapped__", None)
        if inner is None:
            raise RuntimeError(
                f"cannot take a gradient through a '{self.type}' policy: its "
                "predict_action_chunk is not a torch.no_grad wrapper, so there "
                "is nothing to unwrap. Occlusion still works and needs no gradient."
            )
        # ONE wrapper is not enough on pi0.5. `predict_action_chunk` is decorated,
        # and so is `PI05Pytorch.sample_actions` underneath it, so unwrapping only
        # the outer one returns a tensor with no graph and autograd.grad raises
        # "does not require grad" -- which reads as a broken input, not as a
        # second decorator. Both come off, and the sampler is pinned while they
        # are off, because it integrates from noise like diffusion does.
        from actoris_harena.analysis.diffusion import pinned

        with self._grad_through_sampler(), pinned(self):
            return inner(self.policy, dict(batch))

    @contextlib.contextmanager
    def _grad_through_sampler(self):
        """Take ``@torch.no_grad()`` off the inner sampler for one call.

        Bound onto the instance and deleted afterwards, so the class is left
        exactly as it was found -- a policy served in the same process must not
        start building graphs because something asked it for a gradient once.
        """
        model = getattr(self.policy, "model", None)
        sampler = getattr(model, "sample_actions", None)
        unwrapped = getattr(sampler, "__wrapped__", None)
        if model is None or unwrapped is None:
            yield
            return
        import types

        setattr(model, "sample_actions", types.MethodType(unwrapped, model))
        try:
            yield
        finally:
            # Removing the instance attribute uncovers the class's decorated one.
            try:
                delattr(model, "sample_actions")
            except AttributeError:  # pragma: no cover - never bound
                pass

    def image_keys(self) -> "list[str]":
        """The batch keys the cameras arrive under, in the policy's own order."""
        return list(self.cfg.image_features)

    def describe(self) -> str:
        return (
            f"{self.type} on {self.device}: {len(self.cameras)} camera(s), "
            f"{self.n_obs_steps} obs step(s) in, {self.n_action_steps} actions "
            f"out, {self.action_dim}-D"
        )
