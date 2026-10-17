#!/usr/bin/env python
"""DinoViTacFormer: ACT with ViTacFormer's visuo-tactile cross-attention.

ViTacFormer (Heng et al., RSS 2026, arXiv 2506.15953) keeps ACT's conditional
VAE and changes what its encoder reads. Vision and touch are fused by
cross-attention in BOTH directions -- vision tokens query touch, touch tokens
query vision -- and the two outputs are concatenated into the hidden states. A
first pass forecasts future tactile signals from that fusion; a second pass
reads the forecasts as extra touch tokens, fuses again, and decodes the action
chunk. For the first 75 % of training the second pass is given the TRUE future
signals instead (a scheduled-sampling curriculum), and the forecasts after that.

Here the overhead camera is read by a frozen DINOv2 and the four fingertip
cameras by one ResNet-18 trained from scratch, shared by the four. The paper's
end-effector loss is left out: this rig's actions are joint positions.

It subclasses the act_crop config for its tactile camera list and its crop step;
with ``tactile_crop=[1.0,1.0]`` the crop is a no-op, which is how it is run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig

from ..act_crop.configuration_act_crop import HarenaActCropConfig


@PreTrainedConfig.register_subclass("harena_dinovitac")
@dataclass
class HarenaDinovitacConfig(HarenaActCropConfig):
    """HarenaActCropConfig plus the DINOv2 encoder, the fusion and the forecast."""

    #: Images reach the model in [0, 1]; it applies ImageNet statistics itself,
    #: which is what a FROZEN DINOv2 needs -- dataset statistics would shift its
    #: input away from what it was trained on, and nothing could adapt to that.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    #: ACT's ResNet is built by the parent model and then discarded, so it need
    #: not fetch ImageNet weights first.
    pretrained_backbone_weights: str | None = None

    #: The overhead camera's encoder, a Hugging Face DINOv2.
    dino_model_id: str = "facebook/dinov2-small"
    #: False finetunes DINOv2 instead, at ``optimizer_lr`` as ACT finetunes its
    #: ResNet; its input keeps the ImageNet statistics it was pretrained on.
    freeze_dino: bool = True
    #: The size the overhead image is resized to before DINOv2. Both must be
    #: multiples of its 14-pixel patch: 224x294 is 16x21 = 336 tokens, about the
    #: 4:3 of the camera.
    dino_image_size: tuple[int, int] = (224, 294)

    #: Bidirectional cross-attention layers in each of the two fusion stacks.
    #: 0 is plain concatenation of the two token sets, which is the paper's
    #: "w/o CrossAtten" ablation.
    cross_attention_layers: int = 2

    #: False drops the forecast altogether -- its pass, its tokens, its loss and
    #: the curriculum -- so the action pass reads the present fingertips only.
    tactile_forecast: bool = True
    #: Frames ahead whose tactile images are forecast; 10 and 20 are 0.33 and
    #: 0.67 s at the rig's 30 fps. They also become the dataset's observation window.
    tactile_forecast_offsets: tuple[int, ...] = (10, 20)
    #: Each forecast is the future fingertip image area-downsampled to this
    #: (height, width), in RGB -- the "signal" the paper forecasts with L1.
    tactile_forecast_size: tuple[int, int] = (24, 32)
    tactile_forecast_weight: float = 1.0
    #: Training steps for which the action pass reads the TRUE future signals;
    #: after it, the forecasts. The paper switches at 75 % of training, which is
    #: 60000 of ACT's 80000 steps.
    curriculum_steps: int = 60000
    #: Transformer encoder layers in the forecast pass (the action pass uses
    #: ACT's ``n_encoder_layers``).
    forecast_encoder_layers: int = 2

    def __post_init__(self):
        super().__post_init__()
        for side in self.dino_image_size:
            if side % 14:
                raise ValueError(
                    f"dino_image_size must be multiples of DINOv2's 14-pixel patch, "
                    f"got {self.dino_image_size}"
                )
        if self.tactile_forecast and (
            not self.tactile_forecast_offsets or min(self.tactile_forecast_offsets) < 1
        ):
            raise ValueError(
                "tactile_forecast_offsets must name at least one frame ahead, got "
                f"{self.tactile_forecast_offsets}"
            )
        if self.cross_attention_layers < 0:
            raise ValueError("cross_attention_layers cannot be negative")

    def validate_features(self) -> None:
        super().validate_features()
        names = [key.rsplit(".", 1)[-1] for key in self.image_features]
        cameras = set(self.tactile_cameras)
        if not any(n in cameras for n in names) or all(n in cameras for n in names):
            raise ValueError(
                "DinoViTacFormer needs at least one tactile and one non-tactile camera; "
                f"got {list(self.image_features)}"
            )

    @property
    def observation_delta_indices(self) -> "list | None":
        """The present, then each forecast offset; the present alone without a forecast.

        LeRobot applies this to EVERY observation key, so the overhead camera
        and the state arrive with the same time axis; the model reads frame 0
        of them and the later frames of the fingertips only as targets.
        """
        if not self.tactile_forecast:
            return None
        return [0, *self.tactile_forecast_offsets]
