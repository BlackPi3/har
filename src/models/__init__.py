# This source code was written with the assistance of GitHub Copilot autocomplete.
# The author has thoroughly tested and reviewed all code.

"""
Reusable models for human activity recognition.
"""

from .tcn import TCNBlock
from .regressor import Regressor
from .classifier import FeatureExtractor, ActivityClassifier, MmfitEncoder1D, MlpClassifier
from .discriminator import FeatureDiscriminator

__all__ = [
    'TCNBlock',
    'Regressor',
    'FeatureExtractor',
    'ActivityClassifier',
    'MmfitEncoder1D',
    'MlpClassifier',
    'FeatureDiscriminator'
]
