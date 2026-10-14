"""DinoViTacFormer: ACT with frozen DINOv2, a scratch tactile encoder and ViTacFormer's fusion."""

from .configuration_dinovitac import HarenaDinovitacConfig
from .modeling_dinovitac import HarenaDinovitacPolicy
from .processor_dinovitac import make_harena_dinovitac_pre_post_processors

__all__ = [
    "HarenaDinovitacConfig",
    "HarenaDinovitacPolicy",
    "make_harena_dinovitac_pre_post_processors",
]
