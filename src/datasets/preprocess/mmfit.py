"""
MM-Fit accelerometer preprocessing.

Raw pose dumps stored under `datasets/mmfit/wxx/wxx_pose_3d.npy` follow the legacy
format: shape `(3, N, 18)` where the first axis is (x, y, z), the second axis is
time, and the channel axis packs `[frame, joint_0, joint_1, ..., joint_16]`.
Legacy preprocessing injects a timestamp column alongside the frame column,
yielding `(3, M, 19)` via `[frame, timestamp, joints...]`. The code below
replicates that behavior for consistency with older training pipelines: we only
strip the frame column before interpolation so no joint is lost, then reinsert
frame+timestamp metadata ahead of the 17 joints.

Standardizes raw `sw_*_acc.npy` files (per subject) without altering length or
metadata columns. Mean/std are computed over the configured training subjects
so validation/test splits use the same normalization constants.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


@dataclass
class MMFitPreprocessConfig:
    data_dir: Path
    train_subjects: Optional[Sequence[str]] = None
    overwrite: bool = False
    dtype: np.dtype = np.float32
    pose_raw_name: str = "pose_3d.npy"
    pose_aligned_suffix: str = "pose_3d_aligned.npy"


class MMFitPreprocessor:
    def __init__(self, config: MMFitPreprocessConfig):
        self.config = config
        self.acc_mean: Optional[np.ndarray] = None
        self.acc_std: Optional[np.ndarray] = None

    def run(self) -> dict:
        all_subjects = self._list_all_subjects()
        if not all_subjects:
            raise RuntimeError(f"No subjects found under {self.config.data_dir}")

        train_subjects = (
            [s for s in all_subjects if self.config.train_subjects and s.name in self.config.train_subjects]
            if self.config.train_subjects
            else all_subjects
        )
        self._fit_stats(train_subjects)

        processed_files: List[Tuple[str, str]] = []
        for subject in all_subjects:
            acc_files = self._process_subject(subject)
            processed_files.extend(acc_files)
            pose_outputs = self._process_pose(subject, acc_files)
            processed_files.extend(pose_outputs)

        summary = {
            "subjects": [s.name for s in all_subjects],
            "train_subjects": [s.name for s in train_subjects],
            "acc_mean": self.acc_mean.tolist() if self.acc_mean is not None else None,
            "acc_std": self.acc_std.tolist() if self.acc_std is not None else None,
            "processed_files": processed_files,
        }
        stats_path = self.config.data_dir / "mmfit_preprocess_stats.json"
        with stats_path.open("w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # ------------------------------------------------------------------ helpers
    def _list_all_subjects(self) -> List[Path]:
        return sorted(p for p in self.config.data_dir.iterdir() if p.is_dir())

    def _fit_stats(self, subjects: Sequence[Path]) -> None:
        acc_sum = np.zeros(3, dtype=np.float64)
        acc_sq_sum = np.zeros(3, dtype=np.float64)
        total = 0

        for subject in subjects:
            for acc_path in self._acc_files_for_subject(subject):
                vals = self._load_acc_columns(acc_path)
                acc_sum += vals.sum(axis=0)
                acc_sq_sum += (vals ** 2).sum(axis=0)
                total += vals.shape[0]

        if total == 0:
            raise RuntimeError("No accelerometer samples found while fitting stats.")

        mean = acc_sum / total
        var = acc_sq_sum / total - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-12))

        self.acc_mean = mean
        self.acc_std = std

    def _process_subject(self, subject: Path) -> List[Tuple[str, str]]:
        outputs: List[Tuple[str, str]] = []
        for acc_path in self._acc_files_for_subject(subject):
            out_path = acc_path.with_name(acc_path.name.replace("_acc.npy", "_acc_std.npy"))
            if out_path.exists() and not self.config.overwrite:
                continue
            arr = np.load(acc_path)
            std_arr = arr.astype(self.config.dtype).copy()
            std_arr[:, 2:5] = self._standardize(arr[:, 2:5])
            np.save(out_path, std_arr)
            outputs.append((str(acc_path), str(out_path)))
        return outputs

    def _process_pose(self, subject: Path, acc_records: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        outputs: List[Tuple[str, str]] = []
        pose_path = subject / f"{subject.name}_{self.config.pose_raw_name}"
        if not pose_path.exists():
            return outputs
        acc_paths = [Path(raw) for raw, _ in acc_records]
        if not acc_paths:
            acc_paths = list(self._acc_files_for_subject(subject))
            if not acc_paths:
                return outputs
        pose = np.load(pose_path)
        acc = np.load(acc_paths[0])
        aligned = self._align_pose_to_acc(pose, acc)
        norm = self._normalize_pose(aligned)
        aligned_path = subject / f"{subject.name}_{self.config.pose_aligned_suffix}"
        norm_path = subject / f"{subject.name}_pose_3d_norm.npy"
        if aligned_path.exists() and not self.config.overwrite:
            return outputs
        np.save(aligned_path, aligned.astype(self.config.dtype))
        np.save(norm_path, norm.astype(self.config.dtype))
        outputs.append((str(pose_path), str(aligned_path)))
        outputs.append((str(pose_path), str(norm_path)))
        return outputs

    def _standardize(self, data: np.ndarray) -> np.ndarray:
        if self.acc_mean is None or self.acc_std is None:
            raise RuntimeError("Accelerometer stats not fitted.")
        return (data - self.acc_mean[None, :]) / (self.acc_std[None, :] + 1e-8)

    def _acc_files_for_subject(self, subject_dir: Path) -> List[Path]:
        return sorted(subject_dir.glob(f"{subject_dir.name}_sw_*_acc.npy"))

    @staticmethod
    def _load_acc_columns(path: Path) -> np.ndarray:
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError(f"Unexpected accelerometer shape {arr.shape} for {path}")
        return arr[:, 2:5].astype(np.float64)

    def _align_pose_to_acc(self, pose: np.ndarray, acc: np.ndarray) -> np.ndarray:
        # pose shape: (3, frames_pose, channels)
        pose_frames = pose[0, :, 0]
        pose_coords = pose[:, :, 1:]
        acc_frames = acc[:, 0]

        # Build time axes using frame indices (100 Hz assumed)
        frame0 = acc_frames[0]
        pose_time = (pose_frames - frame0) / 100.0
        acc_time = (acc_frames - frame0) / 100.0

        aligned_coords = np.empty((pose_coords.shape[0], len(acc_time), pose_coords.shape[2]), dtype=self.config.dtype)
        for axis in range(pose_coords.shape[0]):
            for joint in range(pose_coords.shape[2]):
                aligned_coords[axis, :, joint] = np.interp(
                    acc_time,
                    pose_time,
                    pose_coords[axis, :, joint],
                    left=pose_coords[axis, 0, joint],
                    right=pose_coords[axis, -1, joint],
                )

        out = np.empty((pose.shape[0], len(acc_time), pose_coords.shape[2] + 2), dtype=self.config.dtype)
        out[:, :, 0] = acc_frames
        out[:, :, 1] = acc_time
        out[:, :, 2:] = aligned_coords
        return out

    def _normalize_pose(self, pose: np.ndarray) -> np.ndarray:
        coords = pose[:, :, 2:]
        midhip_idx = 0  # assumes first joint corresponds to mid-hip after slicing
        reference = coords[:, 0, midhip_idx][:, None, None]
        centered = coords - reference
        return centered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MM-Fit accelerometer preprocessing.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mmfit"))
    parser.add_argument("--train-subjects", nargs="*", default=None, help="Subjects used to compute mean/std.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing *_acc_std.npy files.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _load_train_subjects_from_conf() -> Optional[List[str]]:
    cfg_path = Path(__file__).resolve().parents[3] / "conf" / "data" / "mmfit.yaml"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    subjects = data.get("train_subjects")
    if isinstance(subjects, list) and subjects:
        return subjects
    return None


def main() -> None:
    args = parse_args()
    train_subjects = args.train_subjects or _load_train_subjects_from_conf()
    config = MMFitPreprocessConfig(
        data_dir=args.data_dir,
        train_subjects=train_subjects,
        overwrite=args.overwrite,
    )
    preprocessor = MMFitPreprocessor(config)
    summary = preprocessor.run()
    if not args.quiet:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
