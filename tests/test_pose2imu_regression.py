import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import yaml

from src.datasets.mmfit.constants import ACTIONS
from src.datasets.mmfit.dataset import MMFit
from src.models.regressor import Regressor


REPO_ROOT = Path(__file__).resolve().parents[1]
_TRIAL_NAME = "scenario2_mmfit"
_LEGACY_SCENARIO_NAME = "scenario2"
_BEST_RUN_STEM = "20251109-0055"
_RUN_DIR_CANDIDATES = [
    REPO_ROOT / "experiments" / "best_run" / _TRIAL_NAME / "mmfit" / _BEST_RUN_STEM,
    REPO_ROOT / "experiments" / "best_run" / _LEGACY_SCENARIO_NAME / "mmfit" / _BEST_RUN_STEM,
]
RUN_DIR = next((p for p in _RUN_DIR_CANDIDATES if p.exists()), _RUN_DIR_CANDIDATES[0])
CFG_PATH = RUN_DIR / "resolved_config.yaml"
POSE2IMU_CKPT = RUN_DIR / "checkpoints" / "pose2imu_best.pth"
DATA_ROOT = REPO_ROOT / "datasets" / "mmfit"
POSE2IMU_PLOT_DIR = REPO_ROOT / "tests" / "test_outputs" / "pose2imu"
POSE_WINDOW_PLOT_DIR = REPO_ROOT / "tests" / "test_outputs" / "pose_windows"
NUM_SAMPLES = 10
LABEL_NAMES = {v: k for k, v in ACTIONS.items()}
NON_ACTIVITY_ID = ACTIONS["non_activity"]

_SKIP_REASON = None
if not CFG_PATH.exists():
    _SKIP_REASON = f"missing config: {CFG_PATH}"
elif not POSE2IMU_CKPT.exists():
    _SKIP_REASON = f"missing pose2imu checkpoint: {POSE2IMU_CKPT}"
elif not DATA_ROOT.exists():
    _SKIP_REASON = f"missing dataset root: {DATA_ROOT}"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "missing artifacts")


def _select_device() -> torch.device:
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS device required for this test but is unavailable.")
    return torch.device("mps")


def _read_cfg():
    return yaml.safe_load(CFG_PATH.read_text())


def _load_pose2imu():
    cfg = _read_cfg()
    reg_cfg = cfg["model"]["regressor"]
    window = int(cfg["sensor_window_length"])
    model = Regressor(
        in_ch=reg_cfg.get("input_channels", 3),
        num_joints=reg_cfg.get("num_joints", 3),
        window_length=window,
        joint_hidden_channels=reg_cfg.get("joint_hidden_channels"),
        joint_kernel_sizes=reg_cfg.get("joint_kernel_sizes"),
        joint_dilations=reg_cfg.get("joint_dilations"),
        joint_dropouts=reg_cfg.get("joint_dropouts"),
        temporal_hidden_channels=reg_cfg.get("temporal_hidden_channels"),
        temporal_kernel_size=reg_cfg.get("temporal_kernel_size"),
        temporal_dilation=reg_cfg.get("temporal_dilation"),
        temporal_dropout=reg_cfg.get("temporal_dropout"),
        fc_hidden=reg_cfg.get("fc_hidden"),
        fc_dropout=reg_cfg.get("fc_dropout", 0.0),
        use_batch_norm=reg_cfg.get("use_batch_norm", False),
    )
    device = _select_device()
    state = torch.load(POSE2IMU_CKPT, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return cfg, model, device


def _build_dataset(cfg, subject_id: str) -> MMFit:
    subject_dir = DATA_ROOT / subject_id
    pose_path = subject_dir / f"{subject_id}_{cfg['pose_file']}"
    acc_path = subject_dir / f"{subject_id}_{cfg['acc_file']}"
    labels_path = subject_dir / f"{subject_id}_{cfg['labels_file']}"
    return MMFit(
        pose_file=str(pose_path),
        acc_file=str(acc_path),
        labels_file=str(labels_path),
        sensor_window_length=int(cfg["sensor_window_length"]),
        stride_seconds=cfg.get("stride_seconds"),
        sampling_rate_hz=int(cfg.get("sampling_rate_hz", 100)),
        cluster=bool(cfg.get("cluster", False)),
    )


def _gather_windows(cfg, count: int, seed: int = 0):
    subjects = cfg.get("train_subjects") or []
    if not subjects:
        raise AssertionError("cfg.train_subjects is empty; cannot draw samples")
    rng = random.Random(seed)
    samples = []
    for subject in subjects:
        dataset = _build_dataset(cfg, subject)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        for idx in indices:
            pose, acc, label = dataset[idx]
            if int(label) == NON_ACTIVITY_ID:
                continue
            label_name = LABEL_NAMES.get(int(label), f"label{label}")
            samples.append((pose, acc, label, f"{subject}_{idx}_{label_name}"))
            if len(samples) >= count:
                return samples
    raise AssertionError(f"Only collected {len(samples)} samples; need {count}")


@pytest.fixture(scope="module")
def sampled_windows():
    cfg = _read_cfg()
    return _gather_windows(cfg, count=NUM_SAMPLES, seed=42)


def _save_overlay_plot(real_acc: torch.Tensor, pred_acc: torch.Tensor, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timesteps = range(real_acc.shape[-1])
    axes_labels = ("X", "Y", "Z")
    rows = len(axes_labels) * 2
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(9, 2 * rows))
    for i, axis_name in enumerate(axes_labels):
        real_axis = axes[2 * i]
        pred_axis = axes[2 * i + 1]
        real_axis.plot(timesteps, real_acc[i].cpu().numpy(), color="#1b9e77")
        real_axis.set_ylabel(f"{axis_name} real")
        real_axis.grid(True, linestyle=":", linewidth=0.5)

        pred_axis.plot(timesteps, pred_acc[i].cpu().numpy(), color="#d95f02")
        pred_axis.set_ylabel(f"{axis_name} pred")
        pred_axis.grid(True, linestyle=":", linewidth=0.5)

    axes[-1].set_xlabel("Sample")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_pose_window_plot(pose: torch.Tensor, acc: torch.Tensor, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timesteps = range(acc.shape[-1])
    joint_names = ("Left Shoulder", "Left Elbow", "Left Wrist")
    axes_labels = ("X", "Y", "Z")
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8))

    axes[0].plot(timesteps, acc[0].cpu().numpy(), label="acc X", color="#1b9e77")
    axes[0].plot(timesteps, acc[1].cpu().numpy(), label="acc Y", color="#d95f02")
    axes[0].plot(timesteps, acc[2].cpu().numpy(), label="acc Z", color="#7570b3")
    axes[0].set_ylabel("Accel")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", linewidth=0.5)

    for joint_idx in range(3):
        ax = axes[joint_idx + 1]
        for axis_idx, color in enumerate(("#1b9e77", "#d95f02", "#7570b3")):
            ax.plot(timesteps, pose[axis_idx, joint_idx].cpu().numpy(), color=color, label=f"{axes_labels[axis_idx]}")
        ax.set_ylabel(joint_names[joint_idx])
        ax.grid(True, linestyle=":", linewidth=0.5)

    axes[-1].set_xlabel("Sample")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def test_pose2imu_regressor_visual_regression(sampled_windows):
    _, pose2imu, device = _load_pose2imu()

    pose_batch = torch.stack([pose for pose, _, _, _ in sampled_windows]).to(device)
    acc_batch = torch.stack([acc for _, acc, _, _ in sampled_windows]).to(device)

    with torch.no_grad():
        pred_batch = pose2imu(pose_batch)

    mse_per_sample = torch.mean((pred_batch - acc_batch) ** 2, dim=(1, 2))
    avg_mse = float(mse_per_sample.mean())

    assert not math.isnan(avg_mse)
    assert avg_mse < 12.0, f"Average MSE too high ({avg_mse:.3f}); pose2imu may be untrained"

    for idx, (_, _, _, name) in enumerate(sampled_windows):
        plot_path = POSE2IMU_PLOT_DIR / f"pose2imu_vs_acc_{idx:02d}.png"
        title = f"Sample {idx+1}: {name} | MSE {mse_per_sample[idx]:.2f}"
        _save_overlay_plot(acc_batch[idx], pred_batch[idx], plot_path, title)
        assert plot_path.exists() and plot_path.stat().st_size > 0


def test_pose_and_acc_window_visualization(sampled_windows):
    for idx, (pose, acc, _, name) in enumerate(sampled_windows):
        plot_path = POSE_WINDOW_PLOT_DIR / f"pose_vs_acc_{idx:02d}.png"
        title = f"Sample {idx+1}: {name}"
        _save_pose_window_plot(pose, acc, plot_path, title)
        assert plot_path.exists() and plot_path.stat().st_size > 0
