"""
Data module for HAR project.
Contains dataset classes, loaders, and data utilities.
"""

# Import from new modular structure
from .mmfit import MMFit, build_mmfit_datasets
from .utd import UTDDataset, build_utd_datasets
from .common import RandomStridedSampler, SequentialStridedSampler, BaseHARDataset

# Other dataset classes (to be imported when needed)
# from .mhad import MHADDataset
# from .ntu import NTUDataset  
# from .adv import AdvDataset

__all__ = [
    # MMFit dataset
    'MMFit',
    'build_mmfit_datasets',
    'UTDDataset',
    'build_utd_datasets',
    
    # Common utilities
    'RandomStridedSampler',
    'SequentialStridedSampler', 
    'BaseHARDataset',
    
    # Other datasets (uncomment when ready)
    # 'MHADDataset',
    # 'NTUDataset',
    # 'AdvDataset',
]

# Dataset factory mapping for experiments
DATASET_FACTORIES = {
    'mmfit': build_mmfit_datasets,
    'utd': build_utd_datasets,
    # 'ntu': build_ntu_datasets,    # TODO: Create factory function
}

def get_dataset_factory(dataset_name: str):
    """
    Get the factory function for a dataset.
    
    Args:
        dataset_name: Name of the dataset ('mmfit', 'utd', 'ntu', etc.)
        
    Returns:
        Factory function that takes config and returns (train, val, test) datasets
    """
    if dataset_name not in DATASET_FACTORIES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_FACTORIES.keys())}")
    return DATASET_FACTORIES[dataset_name]
