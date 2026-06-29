"""Speculative decoding engine.

This package implements the exact sampling correction from Leviathan,
Kalman, and Matias, "Fast Inference from Transformers via Speculative
Decoding" (ICML 2023).
"""

from .core import GenerationStats, SpeculativeDecoder, SpeculativeStep
from .models import BigramModel, LanguageModel

__all__ = [
    "BigramModel",
    "GenerationStats",
    "LanguageModel",
    "SpeculativeDecoder",
    "SpeculativeStep",
]
