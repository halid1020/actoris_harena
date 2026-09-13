"""Every policy this rig trains, implemented here rather than in LeRobot.

Importing this package REGISTERS each policy with LeRobot's draccus registry,
which is the whole point of it: ``lerobot.configs.parser.wrap`` loads a package
named by ``--policy.discover_packages_path`` before draccus parses anything, and
``PreTrainedConfig.register_subclass`` fires as a side effect of that import.
After it, ``lerobot.policies.factory.get_policy_class`` resolves our names by
the same route as its own, so one command trains any of them::

    lerobot-train --policy.discover_packages_path=actoris_harena.policies \
                  --policy.type=harena_act ...

LeRobot then derives the rest of the wiring from the config class NAME, purely
mechanically (``policies/factory.py:606``), so the naming is a contract and not
a style:

    actoris_harena.policies/<x>/configuration_<x>.py   So101<X>Config, registered "so101_<x>"
    actoris_harena.policies/<x>/modeling_<x>.py        So101<X>Policy
    actoris_harena.policies/<x>/processor_<x>.py       make_harena_<x>_pre_post_processors

Three of these -- act, diffusion, pi05 -- are PORTS: the upstream module tree
moved here unchanged, so their ``state_dict`` keys are identical to LeRobot's
and every checkpoint already trained still loads. ``test/unit/test_policy_ports.py``
holds them to that. The rest are ours.
"""

from actoris_harena.policies.act.configuration_act import HarenaActConfig
from actoris_harena.policies.act.modeling_act import HarenaActPolicy
from actoris_harena.policies.act_crop.configuration_act_crop import HarenaActCropConfig
from actoris_harena.policies.act_crop.modeling_act_crop import HarenaActCropPolicy
from actoris_harena.policies.diffusion.configuration_diffusion import (
    HarenaDiffusionConfig,
)
from actoris_harena.policies.diffusion.modeling_diffusion import HarenaDiffusionPolicy
from actoris_harena.policies.diffusion_crop.configuration_diffusion_crop import (
    HarenaDiffusionCropConfig,
)
from actoris_harena.policies.diffusion_crop.modeling_diffusion_crop import (
    HarenaDiffusionCropPolicy,
)
from actoris_harena.policies.dreamzero.configuration_dreamzero import (
    HarenaDreamzeroConfig,
)
from actoris_harena.policies.dreamzero.modeling_dreamzero import HarenaDreamzeroPolicy
from actoris_harena.policies.fastwam.configuration_fastwam import HarenaFastwamConfig
from actoris_harena.policies.fastwam.modeling_fastwam import HarenaFastwamPolicy
from actoris_harena.policies.flowmatch.configuration_flowmatch import (
    HarenaFlowmatchConfig,
)
from actoris_harena.policies.flowmatch.modeling_flowmatch import HarenaFlowmatchPolicy
from actoris_harena.policies.pi05.configuration_pi05 import HarenaPi05Config
from actoris_harena.policies.pi05.modeling_pi05 import HarenaPi05Policy
from actoris_harena.policies.pi05_crop.configuration_pi05_crop import (
    HarenaPi05CropConfig,
)
from actoris_harena.policies.pi05_crop.modeling_pi05_crop import HarenaPi05CropPolicy

#: Registered type -> the LeRobot type it was ported from. A checkpoint written
#: by either side of a pair carries the other's name in ``config.json``, and
#: this is what lets one be loaded as the other.
PORTED_FROM = {
    "harena_act": "act",
    "harena_diffusion": "diffusion",
    "harena_pi05": "pi05",
    # Ported from a commit newer than LEROBOT_COMMIT, so its twin does not exist
    # in the installed LeRobot at all -- `matrix.policy_available` refuses a bare
    # `fastwam` row for exactly that reason, while this one runs.
    "harena_fastwam": "fastwam",
}

__all__ = [
    "PORTED_FROM",
    "HarenaActConfig",
    "HarenaActCropConfig",
    "HarenaActCropPolicy",
    "HarenaActPolicy",
    "HarenaDiffusionConfig",
    "HarenaDiffusionCropConfig",
    "HarenaDiffusionCropPolicy",
    "HarenaDiffusionPolicy",
    "HarenaDreamzeroConfig",
    "HarenaDreamzeroPolicy",
    "HarenaFastwamConfig",
    "HarenaFastwamPolicy",
    "HarenaFlowmatchConfig",
    "HarenaFlowmatchPolicy",
    "HarenaPi05Config",
    "HarenaPi05CropConfig",
    "HarenaPi05CropPolicy",
    "HarenaPi05Policy",
]
