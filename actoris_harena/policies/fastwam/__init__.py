"""Ported from ``lerobot.policies.fastwam``; see the package docstring.

Unlike the other three ports this one comes from a commit NEWER than
``LEROBOT_COMMIT`` -- see ``_port.PORT_REF`` for why the pin was not moved to
reach it.

Written by hand rather than ported: upstream's ``__init__`` re-exports
``FastWAMPolicy``, and the port renames that class without adding an alias (the
configuration rewrite adds one; the modeling rewrite deliberately does not).
"""
