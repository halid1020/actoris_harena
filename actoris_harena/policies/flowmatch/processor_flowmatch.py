#!/usr/bin/env python
"""Pre- and post-processing for the flow-matching policy.

The same four steps every policy here uses -- rename, batch, device, normalise --
because none of what makes this policy different happens outside the model.
``lerobot.policies.factory`` finds this by name, so the function's spelling is
part of the contract, not a choice.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_flowmatch import HarenaFlowmatchConfig


def make_harena_flowmatch_pre_post_processors(
    config: HarenaFlowmatchConfig,
    dataset_stats: "dict[str, dict[str, torch.Tensor]] | None" = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


# And under the former name, for a config decoded from an older
# checkpoint that reports the legacy type.
make_so101_flowmatch_pre_post_processors = make_harena_flowmatch_pre_post_processors
