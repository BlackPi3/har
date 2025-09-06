from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader

# ...existing code...
def get_dataloaders(name: str, cfg) -> Dict[str, DataLoader]:
    """
    Return dict with 'train','val','test' DataLoaders for dataset `name`.
    Keep dataset-specific logic here.
    """
    if name == "mmfit":
        from datasets.mmfit_data import build_mmfit_datasets  # dataset factory inside datasets
        train_ds, val_ds, test_ds = build_mmfit_datasets(cfg)
    elif name == "other":
        from datasets.other_data import build_other_datasets
        train_ds, val_ds, test_ds = build_other_datasets(cfg)
    else:
        raise ValueError(name)

    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    return {
        "train": DataLoader(train_ds, shuffle=True, **loader_args),
        "val": DataLoader(val_ds, shuffle=False, **loader_args),
        "test": DataLoader(test_ds, shuffle=False, **loader_args),
    }
# ...existing code...