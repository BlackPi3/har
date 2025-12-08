from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from src.config import get_data_cfg_value
from .dataset import NTUDataset


def build_ntu_datasets(cfg) -> Tuple[Dataset, Dataset, Dataset]:
    data_dir = Path(get_data_cfg_value(cfg, "data_dir", "datasets/ntu"))
    if not data_dir.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        data_dir = (repo_root / data_dir).resolve()
    sampling_rate = int(get_data_cfg_value(cfg, "sampling_rate_hz", 50))
    window = int(get_data_cfg_value(cfg, "sensor_window_length", 100))
    stride = float(get_data_cfg_value(cfg, "stride_seconds", 0.5))
    pad_short = bool(get_data_cfg_value(cfg, "pad_short_clips", True))
    dtype = getattr(cfg, "dtype", None) or torch.float32

    def _normalize_subjects(raw):
        if not raw:
            return None
        return [f"{int(s):03d}" for s in raw]

    train_subjects = _normalize_subjects(get_data_cfg_value(cfg, "train_subjects", None))
    val_subjects = _normalize_subjects(get_data_cfg_value(cfg, "val_subjects", None))
    test_subjects = _normalize_subjects(get_data_cfg_value(cfg, "test_subjects", None))

    if train_subjects is None and val_subjects is None and test_subjects is None:
        all_subj = sorted(
            {p.name[1:] if p.name.lower().startswith("p") else p.name for p in data_dir.glob("p*") if p.is_dir()}
        )
        train_subjects = all_subj
        val_subjects = []
        test_subjects = []

    def _build(subjects):
        if not subjects:
            return None
        return NTUDataset(
            data_dir=data_dir,
            subjects=subjects,
            sampling_rate_hz=sampling_rate,
            window_length=window,
            stride_seconds=stride,
            pad_short_clips=pad_short,
            dtype=dtype,
        )

    return _build(train_subjects), _build(val_subjects), _build(test_subjects)
