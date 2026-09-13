"""Two processor helpers the pinned LeRobot does not ship yet.

FastWAM is newer than `LEROBOT_COMMIT` (3dd19d04) and its processor module calls
`make_default_policy_processor_steps` and `make_policy_processor_pipelines`,
which arrived in `lerobot/processor/factory.py` after that commit. Everything
else FastWAM touches exists at the pin -- MEASURED, name by name -- so this two-
function gap is the entire distance between the port and a working policy.

**Why a shim rather than an edit.** `documents/policy_package.md` says a ported
file that needs a real edit stops being a port. Editing `processor_fastwam.py`
to inline the step list would cost exactly the property the ports exist for: that
`tool/port_policies.py --check` can re-derive every ported file from upstream
byte for byte. So the file is kept verbatim and the *import* is rewritten to
here, which is the same mechanical move `_port._RELATIVE` already makes for
`..pretrained` and `..utils`. The port stays checkable and the compatibility
layer is one named, tested file.

**Why not bump the pin instead.** Moving `LEROBOT_COMMIT` changes act, diffusion
and pi05 underneath every finished checkpoint and invalidates the byte-comparison
the three existing ports rest on -- including the 13 GPU-hours of port-parity
measurement on thanos. A two-function shim is much the smaller change, and the
gate opens by itself when the pin next moves: delete this file and the rewrite
rule beside it.

Both functions are reproduced from upstream's `factory.py` at
`3f2c29ef7`, deliberately line for line rather than improved, so that when the
pin does move the diff is empty and this file can simply go.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@dataclass
class DefaultPolicyProcessorSteps:
    """The canonical processor steps shared by most policies' pre/post pipelines.

    Policies compose these in their own order (step ORDER is a Hub-serialized
    contract and intentionally stays explicit per policy) and interleave their
    custom steps.
    """

    rename_observations: RenameObservationsProcessorStep
    add_batch_dim: AddBatchDimensionProcessorStep
    to_device: DeviceProcessorStep
    normalize: NormalizerProcessorStep
    unnormalize: UnnormalizerProcessorStep
    to_cpu: DeviceProcessorStep


def make_default_policy_processor_steps(
    config: PreTrainedConfig,
    dataset_stats: "dict[str, dict[str, torch.Tensor]] | None" = None,
    *,
    normalizer_device: "torch.device | str | None" = None,
) -> DefaultPolicyProcessorSteps:
    """Construct the canonical policy processor steps from a policy config."""
    return DefaultPolicyProcessorSteps(
        rename_observations=RenameObservationsProcessorStep(rename_map={}),
        add_batch_dim=AddBatchDimensionProcessorStep(),
        to_device=DeviceProcessorStep(device=config.device),
        normalize=NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=normalizer_device,
        ),
        unnormalize=UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        to_cpu=DeviceProcessorStep(device="cpu"),
    )


def make_policy_processor_pipelines(
    input_steps: "list[ProcessorStep]",
    output_steps: "list[ProcessorStep]",
) -> "tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]":
    """Wrap pre/post step lists into the canonical policy pipeline pair.

    Uses the standard pipeline names (which determine the serialized JSON
    filenames on the Hub) and the standard policy-action converters on the
    postprocessor.
    """
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
