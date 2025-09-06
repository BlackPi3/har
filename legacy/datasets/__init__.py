"""
Dataset loaders and preprocessing utilities.
"""

# Import from new modular structure
from .mmfit import MMFit, build_mmfit_datasets
from .common import RandomStridedSampler, SequentialStridedSampler

# Legacy import for backward compatibility (will be removed in future)
from .mmfit_data import unfold  # TODO: Move users to src.inference.unfold_predictions

# Other datasets - keeping existing structure for now
# from .mhad_dataset import MHADDataset
# from .ntu_dataset import NTUDataset
# from .adv_dataset import AdvDataset

# Data loaders - keeping existing structure for now  
# from .mhad_dataloader import mhad_dataloader
# from .ntu_dataloader import ntu_dataloader
# from .adv_dataloader import adv_dataloader

__all__ = [
    # Datasets
    'MMFit',
    # Samplers (from common)
    'RandomStridedSampler',
    'SequentialStridedSampler',
    # Factory functions
    'build_mmfit_datasets',
    # Legacy functions (deprecated)
    'unfold',  # Use src.inference.unfold_predictions instead
]
