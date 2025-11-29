from pathlib import Path
import torch
from torch.utils.data import Dataset

from src.config import get_data_cfg_value
from .dataset import UTDDataset


def build_utd_datasets(cfg) -> tuple[Dataset, Dataset, Dataset]:
    data_dir = Path(get_data_cfg_value(cfg, "data_dir"))
    pose_file = get_data_cfg_value(cfg, "pose_file", "skeleton_aligned.npy")
    acc_file = get_data_cfg_value(cfg, "acc_file", "inertial_std.npy")
    sampling_rate = int(get_data_cfg_value(cfg, "sampling_rate_hz", 50))
    window = int(get_data_cfg_value(cfg, "sensor_window_length", 100))
    stride = float(get_data_cfg_value(cfg, "stride_seconds", 0.5))
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
