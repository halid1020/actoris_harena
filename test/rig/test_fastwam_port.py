"""FastWAM, ported from a commit the pin has never seen.

The other three ports read from the working tree of a checkout pinned at
LEROBOT_COMMIT. FastWAM is not in that commit at all, so this port reads one
directory out of a NAMED newer commit instead. That buys the policy without
moving the pin -- which would change act, diffusion and pi05 underneath every
finished checkpoint and invalidate the byte-comparison the three existing ports
rest on, including the port-parity measurement on thanos.

Two things make it different from the other three, and both are tested here:
its source comes from a ref, and it carries a `wan/` subpackage verbatim. A
third thing is tested because it is the only real gap: the pinned
`lerobot.processor` does not export the two pipeline helpers FastWAM imports, so
the port rewrites that ONE import to a shim rather than editing the file, which
would stop it being a port at all.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_fastwam_port
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PORT = REPO / "actoris_harena" / "policies" / "fastwam"


class PortShapeTest(unittest.TestCase):
    def test_it_is_sourced_from_a_commit_and_not_a_branch(self):
        # `origin/main` moves. A port that silently re-derives from a different
        # upstream on every fetch is not a port.
        from actoris_harena.policies._port import PORT_REF

        ref = PORT_REF["fastwam"]
        self.assertEqual(len(ref), 40, ref)
        self.assertTrue(all(c in "0123456789abcdef" for c in ref), ref)

    def test_the_subpackage_came_with_it(self):
        for name in (
            "__init__",
            "adapters",
            "components",
            "model",
            "modular",
            "video_dit",
        ):
            self.assertTrue((PORT / "wan" / f"{name}.py").is_file(), name)

    def test_the_init_is_ours_not_upstreams(self):
        # Upstream's re-exports FastWAMPolicy, which the port renames without
        # adding an alias -- carrying it verbatim breaks the import outright.
        from actoris_harena.policies._port import PORT_EXTRA

        self.assertNotIn("__init__.py", PORT_EXTRA["fastwam"])
        text = (PORT / "__init__.py").read_text()
        self.assertNotIn("import FastWAMPolicy", text)

    def test_every_file_re_derives(self):
        from actoris_harena.policies._port import port_text, ported_files, read_upstream

        checked = 0
        for relative, ours, name, kind in ported_files():
            if name != "fastwam":
                continue
            try:
                source = read_upstream(name, relative)
            except (FileNotFoundError, OSError):
                self.skipTest("the LeRobot checkout has not fetched the ported commit")
            self.assertEqual(ours.read_text(), port_text(source, name, kind), relative)
            checked += 1
        self.assertGreater(checked, 0)


class TheOneGapTest(unittest.TestCase):
    """The pinned lerobot.processor is missing exactly two names."""

    def test_the_pin_really_does_lack_them(self):
        # If this ever fails, the pin has moved and the shim can be deleted --
        # which is the point of asserting it rather than assuming it.
        import lerobot.processor as processor

        from actoris_harena.policies._port import MISSING_FROM_PIN

        for name in MISSING_FROM_PIN:
            self.assertFalse(
                hasattr(processor, name),
                f"lerobot.processor now exports {name}: delete the shim in "
                "actoris_harena.policies/common/processor_compat.py and the rewrite rule "
                "beside it, then re-run tool/port_policies.py",
            )

    def test_the_shim_supplies_them(self):
        from actoris_harena.policies._port import MISSING_FROM_PIN
        from actoris_harena.policies.common import processor_compat

        for name in MISSING_FROM_PIN:
            self.assertTrue(hasattr(processor_compat, name), name)

    def test_the_ported_file_imports_from_the_shim(self):
        text = (PORT / "processor_fastwam.py").read_text()
        self.assertIn("from ..common.processor_compat import", text)

    def test_the_rewrite_leaves_the_other_names_with_lerobot(self):
        # A rewrite that swallowed the whole import would quietly move names
        # that DO exist upstream into a file we maintain.
        text = (PORT / "processor_fastwam.py").read_text()
        self.assertIn("from lerobot.processor import (", text)
        self.assertIn("ProcessorStepRegistry", text.split("from ..common")[0])

    def test_the_shim_builds_the_same_pipeline_pair_lerobot_would(self):
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature

        from actoris_harena.policies.act.configuration_act import HarenaActConfig
        from actoris_harena.policies.common.processor_compat import (
            make_default_policy_processor_steps,
            make_policy_processor_pipelines,
        )

        config = HarenaActConfig(device="cpu")
        config.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(12,))
        }
        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))
        }
        stats = {
            "observation.state": {"mean": torch.zeros(12), "std": torch.ones(12)},
            "action": {"mean": torch.zeros(12), "std": torch.ones(12)},
        }
        steps = make_default_policy_processor_steps(config, stats)
        pre, post = make_policy_processor_pipelines(
            [
                steps.rename_observations,
                steps.add_batch_dim,
                steps.to_device,
                steps.normalize,
            ],
            [steps.unnormalize, steps.to_cpu],
        )
        # The pipeline NAMES decide the serialized filenames on the Hub, so they
        # are a contract and not a label.
        self.assertEqual(pre.name, "policy_preprocessor")
        self.assertEqual(post.name, "policy_postprocessor")
        self.assertEqual(
            [type(s).__name__ for s in pre.steps],
            [
                "RenameObservationsProcessorStep",
                "AddBatchDimensionProcessorStep",
                "DeviceProcessorStep",
                "NormalizerProcessorStep",
            ],
        )


class RegistrationTest(unittest.TestCase):
    def test_lerobot_resolves_the_policy_by_its_registered_name(self):
        from lerobot.policies.factory import _get_policy_cls_from_policy_name

        import actoris_harena.policies  # noqa: F401

        self.assertEqual(
            _get_policy_cls_from_policy_name("harena_fastwam").__name__,
            "HarenaFastwamPolicy",
        )

    def test_the_whole_module_tree_imports_on_the_pin(self):
        # The question the port exists to answer: does upstream's newer code run
        # against the LeRobot we actually have installed?
        import actoris_harena.policies.fastwam.wan as wan

        for name in ("ActionDiT", "FastWAM", "MoT", "WanVideoDiT", "WanVideoVAE38"):
            self.assertTrue(hasattr(wan, name), name)

    def test_the_untranslated_twin_is_still_refused(self):
        # `fastwam` proper is genuinely absent from the installed LeRobot, and a
        # row naming it must still be refused -- the port does not paper over it.
        from actoris_harena.training.matrix import policy_available

        self.assertFalse(policy_available("fastwam"))
        self.assertTrue(policy_available("harena_fastwam"))

    def test_it_takes_its_twin_s_budget_and_image_size(self):
        from actoris_harena.training.matrix import POLICIES

        port, twin = POLICIES["harena_fastwam"], POLICIES["fastwam"]
        for key in ("steps", "batch", "hours", "image_size"):
            self.assertEqual(port[key], twin[key], key)

    # The test that read this rig's shell driver and checked fastwam took its
    # twin's branch moved to so101_garment with the driver: long_vla_real.sh is
    # that repo's, and a test here cannot read a file that is not in this one.


if __name__ == "__main__":
    unittest.main()
