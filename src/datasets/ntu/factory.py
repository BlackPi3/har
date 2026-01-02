from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from src.config import get_data_cfg_value
from .dataset import NTUDataset


def build_ntu_datasets(cfg) -> Tuple[Dataset, Dataset, Dataset]:
    data_dir_raw = get_data_cfg_value(cfg, "data_dir", None)
    if data_dir_raw is None:
        raise ValueError("data_dir is required for NTU dataset. Set it in conf/data/ntu*.yaml or conf/env/*.yaml")
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        data_dir = (repo_root / data_dir).resolve()
    if not data_dir.exists():
        raise ValueError(f"NTU data_dir does not exist: {data_dir}")
    
    sampling_rate = get_data_cfg_value(cfg, "sampling_rate_hz", None)
    if sampling_rate is None:
        raise ValueError("sampling_rate_hz is required for NTU dataset")
    sampling_rate = int(sampling_rate)
    
    window = get_data_cfg_value(cfg, "sensor_window_length", None)
    if window is None:
        raise ValueError("sensor_window_length is required for NTU dataset")
    window = int(window)
    
    stride = get_data_cfg_value(cfg, "stride_seconds", None)
    if stride is None:
        raise ValueError("stride_seconds is required for NTU dataset")
    stride = float(stride)
    
    pad_short = bool(get_data_cfg_value(cfg, "pad_short_clips", True))
    selected_joints = get_data_cfg_value(cfg, "selected_joints", None)
    if selected_joints is None:
        raise ValueError("selected_joints is required for NTU dataset")
    selected_joints = [int(j) for j in selected_joints]
    
    dtype = getattr(cfg, "dtype", None) or torch.float32

    def _normalize_subjects(raw):
        if not raw:
            return None
        return [f"{int(s):03d}" for s in raw]

    train_subjects = _normalize_subjects(get_data_cfg_value(cfg, "train_subjects", None))
    val_subjects = _normalize_subjects(get_data_cfg_value(cfg, "val_subjects", None))
    test_subjects = _normalize_subjects(get_data_cfg_value(cfg, "test_subjects", None))

    if train_subjects is None:
        raise ValueError("train_subjects is required for NTU dataset")

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
            selected_joints=selected_joints,
            dtype=dtype,
        )

    return _build(train_subjects), _build(val_subjects), _build(test_subjects)
