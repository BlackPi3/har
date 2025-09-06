from typing import Dict
import torch
from torch.utils.data import DataLoader

def get_dataloaders(name: str, cfg) -> Dict[str, DataLoader]:
    """
    Return dict with 'train','val','test' DataLoaders for dataset `name`.
    Keep dataset-specific logic here.
    """
    # Use the new factory system
    if name == "mmfit":
        from src.datasets.mmfit.factory import build_mmfit_datasets
        factory_fn = build_mmfit_datasets
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    train_ds, val_ds, test_ds = factory_fn(cfg)

    # Only use pin_memory on CUDA devices, not on MPS or CPU
    device = getattr(cfg, 'torch_device', torch.device(cfg.device if cfg.device else 'cpu'))
    use_pin_memory = device.type == 'cuda'
    
    # Adjust num_workers for local vs cluster
    num_workers = cfg.num_workers if getattr(cfg, 'cluster', False) else min(cfg.num_workers, 2)
    
    loader_args = dict(batch_size=cfg.batch_size, num_workers=num_workers, pin_memory=use_pin_memory)
    return {
        "train": DataLoader(train_ds, shuffle=True, **loader_args),
        "val": DataLoader(val_ds, shuffle=False, **loader_args),
        "test": DataLoader(test_ds, shuffle=False, **loader_args),
    }
# ...existing code...