from pathlib import Path
import torch
from torch.utils.data import Dataset

from .dataset import UTDMHADDataset


def build_utd_mhad_datasets(cfg) -> tuple[Dataset, Dataset, Dataset]:
    data_dir = Path(getattr(cfg, "data_dir"))
    pose_file = getattr(cfg, "pose_file", "skeleton_aligned.npy")
    acc_file = getattr(cfg, "acc_file", "inertial_std.npy")
    sampling_rate = int(getattr(cfg, "sampling_rate_hz", 50))
    window = int(getattr(cfg, "sensor_window_length", 100))
    stride = float(getattr(cfg, "stride_seconds", 0.5))
    dtype = getattr(cfg, "dtype", None)

    train_subjects = getattr(cfg, "train_subjects", [])
    val_subjects = getattr(cfg, "val_subjects", [])
    test_subjects = getattr(cfg, "test_subjects", [])

    default_dtype = dtype if dtype is not None else torch.float32

    def _build(subjects):
        return UTDMHADDataset(
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
