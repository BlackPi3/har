from typing import Dict
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import get_data_cfg_value

def get_dataloaders(name: str, cfg) -> Dict[str, DataLoader]:
    """
    Return dict with 'train','val','test' DataLoaders for dataset `name`.
    Keep dataset-specific logic here.
    """
    # Use the new factory system
    if name == "mmfit":
        from src.datasets.mmfit.factory import build_mmfit_datasets
        factory_fn = build_mmfit_datasets
    elif name == "utd":
        from src.datasets.utd.factory import build_utd_datasets
        factory_fn = build_utd_datasets
    elif name == "ntu":
        from src.datasets.ntu.factory import build_ntu_datasets
        factory_fn = build_ntu_datasets
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    train_ds, val_ds, test_ds = factory_fn(cfg)

    # Only use pin_memory on CUDA devices, not on MPS or CPU
    device = getattr(cfg, 'torch_device', torch.device(cfg.device if cfg.device else 'cpu'))
    use_pin_memory = device.type == 'cuda'
    
    # Adjust num_workers for local vs cluster
    cfg_num_workers = getattr(cfg, "num_workers", 0) or 0
    num_workers = cfg_num_workers if getattr(cfg, "cluster", False) else min(cfg_num_workers, 2)
    
    batch_size = get_data_cfg_value(cfg, "batch_size")
    if batch_size is None:
        raise ValueError("batch_size is not defined in the config (expected either top-level or under data.*)")

    loader_args = dict(batch_size=int(batch_size), num_workers=num_workers, pin_memory=use_pin_memory)
    if getattr(cfg, "deterministic", False):
        base_seed = getattr(cfg, "seed", None)
        if base_seed is not None:
            def _seed_worker(worker_id: int):
                seed = int(base_seed) + worker_id
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            loader_args["worker_init_fn"] = _seed_worker
            g = torch.Generator()
            g.manual_seed(int(base_seed))
            loader_args["generator"] = g
    trainer_cfg = getattr(cfg, "trainer", None)
    skip_val = bool(getattr(trainer_cfg, "disable_val", False))

    def _as_dataset(name: str, ds, shuffle: bool):
        if ds is None:
            return None
        try:
            n = len(ds)
        except Exception:
            n = 0
        if n == 0:
            if name == "val" and skip_val:
                return None
            raise ValueError(f"{name} dataset is empty; check data_dir/subjects for dataset '{cfg.dataset_name}'.")
        return DataLoader(ds, shuffle=shuffle, **loader_args)

    return {
        "train": _as_dataset("train", train_ds, shuffle=True),
        "val": _as_dataset("val", val_ds, shuffle=False),
        "test": _as_dataset("test", test_ds, shuffle=False),
    }
# ...existing code...
