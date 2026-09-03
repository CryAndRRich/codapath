"""Frozen-backbone feature extraction."""

from .descriptions import description_path, generate_descriptions, load_descriptions
from .visual import DINOv2Extractor, extract_image_features, get_or_extract_features
from .vlm import (
    encode_text_prototypes,
    get_or_extract_vlm_features,
    load_conch,
    text_prototype_cache_paths,
    vlm_feature_cache_paths,
    zero_shot_logits,
)

__all__ = [
    "DINOv2Extractor",
    "extract_image_features",
    "get_or_extract_features",
    "description_path",
    "generate_descriptions",
    "load_descriptions",
    "encode_text_prototypes",
    "get_or_extract_vlm_features",
    "load_conch",
    "text_prototype_cache_paths",
    "vlm_feature_cache_paths",
    "zero_shot_logits",
]
