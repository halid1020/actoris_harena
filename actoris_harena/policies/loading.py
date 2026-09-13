"""Loading a checkpoint into whichever implementation you want.

A checkpoint records the policy type it was trained by, in ``config.json``. That
is normally the right answer, and :func:`ensure_registered` is all a loader needs
-- without it, ``get_policy_class("harena_act")`` cannot resolve a name only this
repo defines, and the failure reads as an unknown policy rather than an unimported
package.

:func:`config_as` is for the other case: reading a checkpoint through the OTHER
member of a ported pair. The ports keep the upstream module tree, so the weights
fit either class; only the type string in ``config.json`` differs. That is what
lets the finished ACT checkpoint be replayed through this repo's copy without
retraining it, and what the equivalence test compares.
"""

from __future__ import annotations

import dataclasses
from typing import Any

_REGISTERED = False


def ensure_registered() -> None:
    """Import this package so its policies exist in LeRobot's registry. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return
    import actoris_harena.policies  # noqa: F401  -- the import IS the registration

    _REGISTERED = True


def config_as(config: Any, target_type: str) -> Any:
    """Rebuild ``config`` as ``target_type``, field for field.

    Both members of a ported pair are the same dataclass under two names, so
    copying the init fields across is exact rather than a best effort. A field
    the target does not declare is a real incompatibility and raises here rather
    than being dropped silently.
    """
    from lerobot.configs import PreTrainedConfig

    ensure_registered()
    target = PreTrainedConfig.get_choice_class(target_type)
    theirs = {f.name for f in dataclasses.fields(target) if f.init}
    ours = {
        f.name: getattr(config, f.name) for f in dataclasses.fields(config) if f.init
    }
    missing = sorted(set(ours) - theirs)
    if missing:
        raise ValueError(
            f"{type(config).__name__} carries {', '.join(missing)}, which "
            f"{target.__name__} does not declare — these are not a ported pair"
        )
    return target(**ours)
