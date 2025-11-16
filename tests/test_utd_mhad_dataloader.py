from pathlib import Path
from types import SimpleNamespace

import torch

from src.datasets.utd_mhad.factory import build_utd_mhad_datasets


def test_utd_mhad_dataloader_shapes():
    cfg = SimpleNamespace(
        data_dir=Path("datasets/UTD_MHAD"),
        pose_file="skeleton_aligned.npy",
        acc_file="inertial_std.npy",
        train_subjects=["s1", "s2"],
        val_subjects=["s3"],
        test_subjects=["s4"],
        sensor_window_length=100,
        stride_seconds=0.5,
        sampling_rate_hz=50,
        dtype=torch.float32,
    )

    train_ds, val_ds, test_ds = build_utd_mhad_datasets(cfg)

    assert len(train_ds) > 0
    pose, acc, label = train_ds[0]
    assert pose.shape == (3, 3, cfg.sensor_window_length)
    assert acc.shape == (3, cfg.sensor_window_length)
    assert isinstance(label, int)

    # Ensure validation/test splits are also constructed
    assert len(val_ds) >= 0
    assert len(test_ds) >= 0
