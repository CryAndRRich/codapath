"""Frozen-backbone feature extraction."""

from .visual import DINOv2Extractor, extract_image_features, get_or_extract_features

__all__ = ["DINOv2Extractor", "extract_image_features", "get_or_extract_features"]
