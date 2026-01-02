from pathlib import Path
import torch
from torch.utils.data import Dataset

from src.config import get_data_cfg_value
from .dataset import UTDDataset


def build_utd_datasets(cfg) -> tuple[Dataset, Dataset, Dataset]:
    data_dir_raw = get_data_cfg_value(cfg, "data_dir", None)
    if data_dir_raw is None:
        raise ValueError("data_dir is required for UTD dataset. Set it in conf/data/utd.yaml or conf/env/*.yaml")
    data_dir = Path(data_dir_raw)
    if not data_dir.exists():
        raise ValueError(f"UTD data_dir does not exist: {data_dir}")
    
    pose_file = get_data_cfg_value(cfg, "pose_file", None)
    if pose_file is None:
        raise ValueError("pose_file is required for UTD dataset")
    acc_file = get_data_cfg_value(cfg, "acc_file", None)
    if acc_file is None:
        raise ValueError("acc_file is required for UTD dataset")
    
    sampling_rate = get_data_cfg_value(cfg, "sampling_rate_hz", None)
    if sampling_rate is None:
        raise ValueError("sampling_rate_hz is required for UTD dataset")
    sampling_rate = int(sampling_rate)
    
    window = get_data_cfg_value(cfg, "sensor_window_length", None)
    if window is None:
        raise ValueError("sensor_window_length is required for UTD dataset")
    window = int(window)
    
    stride = get_data_cfg_value(cfg, "stride_seconds", None)
    if stride is None:
        raise ValueError("stride_seconds is required for UTD dataset")
    stride = float(stride)
    dtype = getattr(cfg, "dtype", None)

    train_subjects = get_data_cfg_value(cfg, "train_subjects", [])
    val_subjects = get_data_cfg_value(cfg, "val_subjects", [])
    test_subjects = get_data_cfg_value(cfg, "test_subjects", [])

    default_dtype = dtype if dtype is not None else torch.float32

    def _build(subjects):
        return UTDDataset(
            data_dir=data_dir,
            subjects=subjects,
            pose_suffix=pose_file,
            acc_suffix=acc_file,
            sensor_window_length=window,
            stride_seconds=stride,
            sampling_rate_hz=sampling_rate,
            dtype=default_dtype,
        )

    return _build(train_subjects), _build(val_subjects), _build(test_subjects)
