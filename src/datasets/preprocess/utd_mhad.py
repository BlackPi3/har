"""
UTD-MHAD dataset preprocessor.

This script converts the raw subject folders into standardized pose/IMU
artifacts that align skeleton clips with the corresponding inertial windows,
drop the rotational channels, and normalize both modalities with consistent
statistics.  It purposefully avoids upsampling to MM-Fit-style 400-sample
windows; instead, each clip keeps its native duration (roughly 107–326 frames).

Usage (from repo root):

    python -m src.datasets.preprocess.utd_mhad --data-dir datasets/UTD_MHAD
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


# Joint indices follow Kinect ordering used in legacy scripts.
HEAD_IDX = 0
MID_HIP_IDX = 3


@dataclass
class UTDMHADPreprocessConfig:
    data_dir: Path
    subjects: Optional[Sequence[str]] = None
    train_subjects: Optional[Sequence[str]] = None
    imu_sampling_rate_hz: float = 50.0
    skeleton_sampling_rate_hz: float = 30.0
    normalization_window_sec: float = 2.0
    inertial_raw_suffix: str = "inertial.npy"
    skeleton_raw_suffix: str = "skeleton.npy"
    inertial_output_suffix: str = "inertial_std.npy"
    skeleton_norm_suffix: str = "skeleton_norm.npy"
    skeleton_output_suffix: str = "skeleton_aligned.npy"
    dtype: np.dtype = np.float32
    overwrite: bool = True
    verbose: bool = True


@dataclass
class _Clip:
    subject: str
    action: str
    trial: str
    prefix: str
    inertial_path: Path
    skeleton_path: Path


@dataclass
class _DatasetStats:
    lengths: List[int] = field(default_factory=list)
    actions: set = field(default_factory=set)
    subjects: set = field(default_factory=set)
    acc_mean: Optional[np.ndarray] = None
    acc_std: Optional[np.ndarray] = None


class UTDMHADPreprocessor:
    def __init__(self, config: UTDMHADPreprocessConfig):
        self.config = config
        self.stats = _DatasetStats()

    # --------------------------------------------------------------------- API
    def run(self) -> Dict[str, float]:
        clips = self._collect_clips()
        if not clips:
            raise RuntimeError(f"No clips found under {self.config.data_dir}")

        self._fit_acc_stats(clips)

        for clip in clips:
            acc = self._process_inertial(clip.inertial_path)
            skel = self._process_skeleton(clip.skeleton_path)
            self._persist_clip(clip, acc, skel)
            self._record_stats(clip, acc_length=acc.shape[1])

        summary = self._finalize()
        if self.config.verbose:
            print(json.dumps(summary, indent=2))
        return summary

    # ------------------------------------------------------------------ Helpers
    def _collect_clips(self) -> List[_Clip]:
        cfg = self.config
        data_dir = cfg.data_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"UTD-MHAD directory not found: {data_dir}")

        subjects = cfg.subjects
        if subjects is None:
            subjects = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("s"))

        clips: List[_Clip] = []
        inertial_suffix = cfg.inertial_raw_suffix
        skeleton_suffix = cfg.skeleton_raw_suffix
        inertial_glob = f"a*_*_*_{inertial_suffix}"

        for subject in subjects:
            subject_dir = data_dir / subject
            if not subject_dir.is_dir():
                continue
            for inertial_path in sorted(subject_dir.glob(inertial_glob)):
                prefix = "_".join(inertial_path.stem.split("_")[:3])
                skeleton_path = subject_dir / f"{prefix}_{skeleton_suffix}"
                if not skeleton_path.exists():
                    if self.config.verbose:
                        print(f"[warn] Missing skeleton for {inertial_path.name}; skipping")
                    continue
                action, _, trial = prefix.split("_")
                clips.append(
                    _Clip(
                        subject=subject,
                        action=action,
                        trial=trial,
                        prefix=prefix,
                        inertial_path=inertial_path,
                        skeleton_path=skeleton_path,
                    )
                )
        if self.config.verbose:
            print(f"[utd_mhad] Collected {len(clips)} clips across {len(subjects)} subjects")
        return clips

    def _fit_acc_stats(self, clips: Sequence[_Clip]) -> None:
        train_subjects = set(self.config.train_subjects) if self.config.train_subjects else None
        acc_bank: List[np.ndarray] = []
        for clip in clips:
            if train_subjects and clip.subject not in train_subjects:
                continue
            acc = self._load_inertial_raw(clip.inertial_path)
            acc_bank.append(acc[:, :3].T)  # shape (3, T)

        if not acc_bank:
            raise RuntimeError("No training clips found to estimate accelerometer statistics.")

        concat = np.concatenate(acc_bank, axis=1)
        self.stats.acc_mean = concat.mean(axis=1)
        self.stats.acc_std = concat.std(axis=1)

    # ----------------------------------------------------------------- Loading
    def _load_inertial_raw(self, path: Path) -> np.ndarray:
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Unexpected inertial array shape {arr.shape} for {path}")
        # Raw inertial files are shaped (N, 6) with N = clip length. The first 3 columns are accelerometer.
        return arr.astype(self.config.dtype, copy=False)

    def _load_skeleton_raw(self, path: Path) -> np.ndarray:
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected skeleton array shape {arr.shape} for {path}")
        # Raw files are (joints, channels, T). Convert to (3, joints, T).
        return np.transpose(arr, (1, 0, 2)).astype(self.config.dtype, copy=False)

    # ----------------------------------------------- Modality-specific process
    def _process_inertial(self, path: Path) -> np.ndarray:
        acc = self._load_inertial_raw(path)[:, :3].T  # -> (3, T)
        mean = self.stats.acc_mean
        std = self.stats.acc_std
        if mean is None or std is None:
            raise RuntimeError("Accelerometer statistics not fitted before processing.")
        acc = (acc - mean[:, None]) / (std[:, None] + 1e-8)
        return acc

    def _process_skeleton(self, path: Path) -> np.ndarray:
        skel = self._load_skeleton_raw(path)
        skel = self._normalize_skeleton(skel)
        # Alignment to IMU timeline will be reintroduced once we finalize the
        # normalization scheme. For now, keep the native pose frame rate.
        # skel = self._align_skeleton(skel, target_length)
        return skel

    def _align_skeleton(self, skel: np.ndarray, target_length: int) -> np.ndarray:
        if skel.shape[2] == target_length:
            return skel

        skel_ts = self._time_axis(skel.shape[2], self.config.skeleton_sampling_rate_hz)
        imu_ts = self._time_axis(target_length, self.config.imu_sampling_rate_hz)

        aligned = np.empty((skel.shape[0], skel.shape[1], target_length), dtype=self.config.dtype)
        for axis in range(skel.shape[0]):
            for joint in range(skel.shape[1]):
                aligned[axis, joint, :] = np.interp(imu_ts, skel_ts, skel[axis, joint, :])
        return aligned

    def _normalize_skeleton(self, skel: np.ndarray) -> np.ndarray:
        midhip = skel[:, MID_HIP_IDX, :]
        head = skel[:, HEAD_IDX, :]

        # Subtract a reference mid-hip position (first frame) to preserve curve shape.
        reference_midhip = midhip[:, 0][:, None, None]
        centered = skel - reference_midhip

        # Global scale per clip (median neck-hip distance) avoids per-frame distortions.
        distances = np.linalg.norm(head - midhip, axis=0)
        distances = np.clip(distances, a_min=1e-4, a_max=None)
        scale = float(np.median(distances))
        scale = max(scale, 1e-3)

        normalized = centered / scale
        return normalized

    # ----------------------------------------------------------------- Utility
    @staticmethod
    def _time_axis(length: int, rate: float) -> np.ndarray:
        if length <= 1 or rate <= 0:
            return np.zeros(length, dtype=np.float32)
        duration = (length - 1) / rate
        return np.linspace(0.0, duration, num=length, dtype=np.float32)

    def _persist_clip(self, clip: _Clip, acc: np.ndarray, skel: np.ndarray) -> None:
        out_acc = clip.inertial_path.with_name(f"{clip.prefix}_{self.config.inertial_output_suffix}")
        out_skel_norm = clip.skeleton_path.with_name(f"{clip.prefix}_{self.config.skeleton_norm_suffix}")

        if out_acc.exists() and self.config.verbose and self.config.overwrite:
            print(f"[overwrite] {out_acc}")
        np.save(out_acc, acc.astype(self.config.dtype))

        if out_skel_norm.exists() and self.config.verbose and self.config.overwrite:
            print(f"[overwrite] {out_skel_norm}")
        np.save(out_skel_norm, skel.astype(self.config.dtype))

        # Alignment output will be added later when the interpolation step returns.
        # out_skel_aligned = clip.skeleton_path.with_name(f"{clip.prefix}_{self.config.skeleton_output_suffix}")
        # np.save(out_skel_aligned, aligned_skel.astype(self.config.dtype))

    def _record_stats(self, clip: _Clip, acc_length: int) -> None:
        self.stats.lengths.append(acc_length)
        self.stats.subjects.add(clip.subject)
        self.stats.actions.add(clip.action)

    def _finalize(self) -> Dict[str, float]:
        lengths = np.asarray(self.stats.lengths)
        summary = {
            "num_clips": int(len(lengths)),
            "subjects": sorted(self.stats.subjects),
            "actions": sorted(self.stats.actions),
            "len_min": float(lengths.min()) if lengths.size else None,
            "len_max": float(lengths.max()) if lengths.size else None,
            "len_mean": float(lengths.mean()) if lengths.size else None,
            "len_median": float(np.median(lengths)) if lengths.size else None,
            "acc_mean": self.stats.acc_mean.tolist() if self.stats.acc_mean is not None else None,
            "acc_std": self.stats.acc_std.tolist() if self.stats.acc_std is not None else None,
        }
        stats_path = self.config.data_dir / "utd_mhad_preprocess_stats.json"
        with stats_path.open("w") as f:
            json.dump(summary, f, indent=2)
        return summary


def _load_train_subjects_from_conf() -> Optional[Sequence[str]]:
    """
    Read train_subjects from conf/data/mhad.yaml to keep preprocessing aligned
    with the dataset config used by the training pipeline.
    """
    cfg_path = Path(__file__).resolve().parents[3] / "conf" / "data" / "mhad.yaml"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open("r") as f:
            data = yaml.safe_load(f) or {}
        subjects = data.get("train_subjects")
        if isinstance(subjects, list) and subjects:
            return subjects
    except Exception as exc:
        print(f"[warn] Could not read train subjects from {cfg_path}: {exc}")
    return None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UTD-MHAD preprocessing pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path(Path(__file__).resolve().parents[3] / "datasets"/"UTD_MHAD"))
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subset of subjects (e.g., s1 s2).")
    parser.add_argument("--train-subjects", nargs="*", default=None, help="Subjects used to compute accelerometer stats.")
    parser.add_argument("--imu-sampling-rate", type=float, default=50.0)
    parser.add_argument("--skeleton-sampling-rate", type=float, default=30.0)
    parser.add_argument("--normalization-window-sec", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_subjects = args.train_subjects or _load_train_subjects_from_conf()
    config = UTDMHADPreprocessConfig(
        data_dir=args.data_dir,
        subjects=args.subjects,
        train_subjects=train_subjects,
        imu_sampling_rate_hz=args.imu_sampling_rate,
        skeleton_sampling_rate_hz=args.skeleton_sampling_rate,
        normalization_window_sec=args.normalization_window_sec,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )
    preprocessor = UTDMHADPreprocessor(config)
    preprocessor.run()


if __name__ == "__main__":
    main()
