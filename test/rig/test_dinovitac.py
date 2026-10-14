"""DinoViTacFormer: ViTacFormer's visuo-tactile cross-attention on ACT.

A tiny random DINOv2 stands in for the pretrained one, so nothing is downloaded.
The tests hold the four things the method rests on: DINOv2 stays frozen, the
fusion runs both ways (and is plain concatenation with no layers), the forecast
is trained on the right frames, and the curriculum switches from the true future
to the forecast -- and never reads the future outside training.

Run:  PYTHONPATH=.:src python -m unittest test.rig.test_dinovitac
"""

from __future__ import annotations

import tempfile
import unittest

import torch

STATE = "observation.state"
CENTRAL = "observation.images.central"
TACTILE = (
    "observation.images.left_arm_left_gripper",
    "observation.images.right_arm_left_gripper",
)
T = 3  # the present and two forecast offsets


def tiny_dino():
    from transformers import Dinov2Config, Dinov2Model

    return Dinov2Model(
        Dinov2Config(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            image_size=28,
            patch_size=14,
        )
    )


def make_config(**overrides):
    from lerobot.configs.types import FeatureType, PolicyFeature

    from actoris_harena.policies.dinovitac.configuration_dinovitac import (
        HarenaDinovitacConfig,
    )

    fields = dict(
        device="cpu",
        tactile_crop=(1.0, 1.0),
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        chunk_size=4,
        n_action_steps=4,
        dino_image_size=(28, 42),
        tactile_forecast_size=(4, 4),
        cross_attention_layers=1,
        forecast_encoder_layers=1,
        curriculum_steps=2,
        dropout=0.0,
    )
    fields.update(overrides)
    config = HarenaDinovitacConfig(**fields)
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))
    config.input_features = {
        STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
        CENTRAL: image,
        **{key: image for key in TACTILE},
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))
    }
    return config


def make_policy(**overrides):
    from actoris_harena.policies.dinovitac.modeling_dinovitac import (
        HarenaDinovitacPolicy,
    )

    torch.manual_seed(0)
    return HarenaDinovitacPolicy(make_config(**overrides), dino=tiny_dino())


def training_batch(seed=0):
    torch.manual_seed(seed)
    batch = {
        STATE: torch.randn(2, T, 12),
        CENTRAL: torch.rand(2, T, 3, 32, 32),
        "action": torch.randn(2, 4, 12),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }
    for key in TACTILE:
        batch[key] = torch.rand(2, T, 3, 32, 32)
        batch[f"{key}_is_pad"] = torch.zeros(2, T, dtype=torch.bool)
    return batch


def present(batch):
    """The single-frame observation a rollout hands in."""
    return {
        key: (
            value[:, 0]
            if key == STATE or key.startswith("observation.images")
            else value
        )
        for key, value in batch.items()
        if not key.endswith("_is_pad")
    }


def model_out(policy, batch):
    from lerobot.utils.constants import OBS_IMAGES

    batch = dict(batch)
    batch[OBS_IMAGES] = [batch[k] for k in policy.config.image_features]
    return policy.model(batch)


class ConfigTest(unittest.TestCase):
    def test_the_window_is_the_present_then_the_offsets(self):
        self.assertEqual(make_config().observation_delta_indices, [0, 10, 20])

    def test_images_are_left_unnormalised_for_the_model_s_own_statistics(self):
        self.assertEqual(
            make_config().normalization_mapping["VISUAL"].value, "IDENTITY"
        )

    def test_a_size_off_the_patch_grid_is_refused(self):
        with self.assertRaises(ValueError):
            make_config(dino_image_size=(224, 300))

    def test_touch_alone_is_refused(self):
        from lerobot.configs.types import FeatureType, PolicyFeature

        config = make_config()
        image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))
        config.input_features = {
            STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
            **{key: image for key in TACTILE},
        }
        with self.assertRaises(ValueError):
            config.validate_features()


class TrainingTest(unittest.TestCase):
    def test_a_window_trains_every_part_but_dinov2(self):
        policy = make_policy()
        policy.train()
        loss, logged = policy.forward(training_batch())
        self.assertEqual(set(logged), {"l1_loss", "kld_loss", "tactile_l1"})
        loss.backward()

        model = policy.model
        self.assertFalse(model.dino.training)
        self.assertTrue(all(p.grad is None for p in model.dino.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.dino.parameters()))
        for part in (
            model.vision_proj,
            model.tactile_backbone,
            model.fusion_forecast,
            model.fusion_action,
            model.forecast_head,
            model.future_proj,
        ):
            grads = [p.grad for p in part.parameters() if p.requires_grad]
            self.assertTrue(grads and all(g is not None for g in grads), part)

    def test_dinov2_is_not_handed_to_the_optimiser(self):
        policy = make_policy()
        optimised = {id(p) for g in policy.get_optim_params() for p in g["params"]}
        self.assertFalse(optimised & {id(p) for p in policy.model.dino.parameters()})

    def test_a_padded_future_frame_is_not_a_target(self):
        policy = make_policy()
        policy.train()
        batch = training_batch()
        _, _, extras = model_out(policy, batch)
        self.assertTrue(extras["valid"].all())

        batch[f"{TACTILE[0]}_is_pad"][0, 2] = True
        _, _, extras = model_out(policy, batch)
        # Fingertip-major: fingertip 0's second offset, for sample 0.
        self.assertFalse(extras["valid"][0, 1])
        self.assertEqual(int((~extras["valid"]).sum()), 1)

    def test_the_target_is_the_future_frame_downsampled(self):
        policy = make_policy()
        policy.train()
        batch = training_batch()
        _, _, extras = model_out(policy, batch)
        expected = torch.nn.functional.adaptive_avg_pool2d(
            batch[TACTILE[1]][:, 2], (4, 4)
        )
        torch.testing.assert_close(extras["target"][:, 3], expected.flatten(1))


class CurriculumTest(unittest.TestCase):
    def actions(self, policy, batch, step):
        policy.model.train_steps.fill_(step)
        torch.manual_seed(1)  # the same latent draw every time
        return model_out(policy, batch)[0]

    def test_the_truth_first_then_the_forecast(self):
        policy = make_policy()
        policy.train()
        batch = training_batch()
        changed = training_batch()
        for key in TACTILE:  # only the FUTURE fingertip frames differ
            changed[key][:, 1:] = torch.rand_like(changed[key][:, 1:])

        def with_head_perturbed(step):
            original = policy.model.forecast_head.bias.detach().clone()
            policy.model.forecast_head.bias.data += 5.0
            try:
                return self.actions(policy, batch, step)
            finally:
                policy.model.forecast_head.bias.data.copy_(original)

        # Truth phase: the forecast head cannot reach the actions, the future can.
        early = self.actions(policy, batch, 0)
        torch.testing.assert_close(early, with_head_perturbed(0))
        self.assertFalse(torch.allclose(early, self.actions(policy, changed, 0)))
        # Forecast phase: the reverse.
        late = self.actions(policy, batch, 2)
        torch.testing.assert_close(late, self.actions(policy, changed, 2))
        self.assertFalse(torch.allclose(late, with_head_perturbed(2)))

    def test_only_a_training_pass_advances_it(self):
        policy = make_policy()
        policy.train()
        model_out(policy, training_batch())
        model_out(policy, training_batch())
        policy.eval()
        with torch.no_grad():
            model_out(policy, training_batch())
        self.assertEqual(int(policy.model.train_steps), 2)

    def test_evaluation_never_reads_the_future(self):
        policy = make_policy()
        policy.eval()
        batch = training_batch()
        changed = training_batch()
        for key in TACTILE:
            changed[key][:, 1:] = torch.rand_like(changed[key][:, 1:])
        with torch.no_grad():
            a, _, extras = model_out(policy, batch)
            b, _, _ = model_out(policy, changed)
        torch.testing.assert_close(a, b)
        self.assertNotIn("target", extras)

    def test_the_phase_survives_a_save_and_load(self):
        from actoris_harena.policies.dinovitac.modeling_dinovitac import (
            HarenaDinovitacPolicy,
        )

        policy = make_policy()
        policy.model.train_steps.fill_(7)
        with tempfile.TemporaryDirectory() as tmp:
            policy.save_pretrained(tmp)
            loaded = HarenaDinovitacPolicy.from_pretrained(tmp, dino=tiny_dino())
        self.assertEqual(int(loaded.model.train_steps), 7)


class FusionTest(unittest.TestCase):
    def tokens(self, layers):
        from actoris_harena.policies.dinovitac.modeling_dinovitac import (
            CrossModalFusion,
        )

        torch.manual_seed(0)
        config = make_config(cross_attention_layers=layers)
        fusion = CrossModalFusion(config, layers).eval()
        v, t = torch.randn(6, 2, 32), torch.randn(10, 2, 32)
        vp, tp = torch.randn(6, 1, 32), torch.randn(10, 1, 32)
        return fusion, v, t, vp, tp

    def test_no_layers_is_plain_concatenation(self):
        fusion, v, t, vp, tp = self.tokens(0)
        tokens, pos = fusion(v, t, vp, tp)
        torch.testing.assert_close(tokens, torch.cat([v, t]))
        torch.testing.assert_close(pos, torch.cat([vp, tp]))

    def test_each_modality_attends_to_the_other(self):
        fusion, v, t, vp, tp = self.tokens(1)
        base, _ = fusion(v, t, vp, tp)
        touched, _ = fusion(v, t + 1.0, vp, tp)
        seen, _ = fusion(v + 1.0, t, vp, tp)
        # Changing touch moves the VISION tokens (the first 6), and vice versa.
        self.assertFalse(torch.allclose(base[:6], touched[:6]))
        self.assertFalse(torch.allclose(base[6:], seen[6:]))


class ServingTest(unittest.TestCase):
    def test_a_rollout_plans_from_single_frames(self):
        policy = make_policy()
        policy.reset()
        action = policy.select_action(present(training_batch()))
        self.assertEqual(tuple(action.shape), (2, 12))
        chunk = policy.predict_action_chunk(present(training_batch()))
        self.assertEqual(tuple(chunk.shape), (2, 4, 12))

    def test_registered_and_in_the_run_matrix(self):
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        import actoris_harena.policies  # noqa: F401  (registers)
        from actoris_harena.policies.dinovitac import (
            HarenaDinovitacPolicy,
            make_harena_dinovitac_pre_post_processors,
        )
        from actoris_harena.training.matrix import POLICIES

        self.assertIs(get_policy_class("harena_dinovitac"), HarenaDinovitacPolicy)
        self.assertEqual(
            PreTrainedConfig.get_choice_class("harena_dinovitac").__name__,
            "HarenaDinovitacConfig",
        )
        self.assertEqual(POLICIES["harena_dinovitac"]["steps"], 80000)
        pre, _ = make_harena_dinovitac_pre_post_processors(make_config())
        self.assertEqual(type(pre.steps[0]).__name__, "HarenaTactileCropProcessorStep")


if __name__ == "__main__":
    unittest.main()
