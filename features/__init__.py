"""Frozen-backbone feature extraction."""

from .visual import DINOv2Extractor, extract_image_features, get_or_extract_features
from .vlm import (
    assert_class_order_matches_prompts,
    encode_text_prototypes,
    get_or_extract_vlm_features,
    load_conch,
    load_official_conch_prompts,
    text_prototype_cache_paths,
    vlm_feature_cache_paths,
    zero_shot_logits,
)

__all__ = [
    "DINOv2Extractor",
    "extract_image_features",
    "get_or_extract_features",
    "assert_class_order_matches_prompts",
    "encode_text_prototypes",
    "get_or_extract_vlm_features",
    "load_conch",
    "load_official_conch_prompts",
    "text_prototype_cache_paths",
    "vlm_feature_cache_paths",
    "zero_shot_logits",
]
