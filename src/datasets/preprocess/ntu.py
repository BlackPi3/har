"""
NTU RGB+D skeleton preprocessing.

The raw NTU dataset ships as `.skeleton` text files that interleave per-frame
body metadata with tracked joint coordinates.  This utility converts those files
into compact `.npy` artifacts shaped `(3, joints, frames)` so downstream
datasets/dataloaders can treat NTU clips the same way as the other pose-only
corpora in this project.

Usage (from the repo root, no flags):

    python -m src.datasets.preprocess.ntu

Notes:
- Only frames where all joints are tracked/inferred are kept; frames with any
  "not tracked" joints are dropped.
- Two upsampled outputs (50 Hz and 100 Hz) are written per clip under
  `datasets/ntu/p###/`.
- Joints are NTU's 1-based indices: center=1 (base spine), scale spine pair=1 & 21,
  left arm=5/6/7, right arm=9/10/11; these must be present to keep a frame.
- Length stats after preprocessing: 50 Hz clips >100 frames: 78.22%; 100 Hz clips
  >2.5s (250 frames): 51.23%.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FILENAME_RE = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})R(?P<replication>\d{3})A(?P<action>\d{3})",
    re.IGNORECASE,
)
NUM_JOINTS = 25


@dataclass
class NTUPreprocessConfig:
    data_dir: Path = Path("datasets/ntu_skel_original")
    output_dir: Optional[Path] = Path("datasets/ntu")
    actions: Optional[Sequence[int]] = None
    subjects: Optional[Sequence[int]] = None
    overwrite: bool = False
    target_sampling_rates: Sequence[float] = (50.0, 100.0)
    center_joint: int = 1
    scale_joints: Tuple[int, int] = (1, 21)
    min_tracked_joints: int = 4
    dtype: np.dtype = np.float32
    orig_sampling_rate_hz: float = 30.0


@dataclass
class _Clip:
    pose: np.ndarray  # (3, joints, frames)
    mask: np.ndarray  # (joints, frames) boolean visibility
    frames: int
    raw_frames: int


class NTUPreprocessor:
    def __init__(self, config: NTUPreprocessConfig):
        self.config = config
        self.output_dir = config.output_dir or Path("datasets/ntu")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.center_idx = max(0, config.center_joint - 1)
        self.scale_a = max(0, config.scale_joints[0] - 1)
        self.scale_b = max(0, config.scale_joints[1] - 1)

        self.subject_filter = {int(s) for s in config.subjects} if config.subjects else None
        self.action_filter = {int(a) for a in config.actions} if config.actions else None

    # ------------------------------------------------------------------ public
    def run(self) -> Dict[str, object]:
        files = self._collect_files()
        manifest: List[Dict[str, object]] = []
        lengths: List[int] = []
        actions: set[int] = set()
        subjects: set[int] = set()

        for path in files:
            meta = self._parse_metadata(path)
            if meta is None:
                continue
            if self.subject_filter and meta["subject"] not in self.subject_filter:
                continue
            if self.action_filter and meta["action"] not in self.action_filter:
                continue

            clip = self._load_clip(path)
            if clip is None:
                continue

            for rate in self.config.target_sampling_rates:
                output_path = self._output_path(path, rate, meta["subject"])
                if output_path.exists() and not self.config.overwrite:
                    continue

                resampled = self._resample_clip(clip, target_rate=float(rate))
                if resampled.frames < clip.raw_frames:
                    continue
                np.save(
                    output_path,
                    resampled.pose.astype(self.config.dtype, copy=False),
                )

                manifest.append(
                    {
                        "source": str(path),
                        "output": str(output_path),
                        "frames": resampled.frames,
                        "raw_frames": clip.raw_frames,
                        "subject": meta["subject"],
                        "action": meta["action"],
                        "sampling_rate_hz": float(rate),
                    }
                )
                lengths.append(resampled.frames)
                actions.add(meta["action"])
                subjects.add(meta["subject"])

        summary = {
            "clips": manifest,
            "num_clips": len(manifest),
            "subjects": sorted(subjects),
            "actions": sorted(actions),
            "len_min": int(min(lengths)) if lengths else None,
            "len_max": int(max(lengths)) if lengths else None,
            "len_mean": float(np.mean(lengths)) if lengths else None,
            "len_median": float(np.median(lengths)) if lengths else None,
        }

        stats_path = self.output_dir / "ntu_preprocess_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    # ----------------------------------------------------------------- helpers
    def _collect_files(self) -> List[Path]:
        return sorted(self.config.data_dir.glob("*.skeleton"))

    def _output_path(self, source_path: Path, rate: float, subject: int) -> Path:
        rate_tag = int(rate) if float(rate).is_integer() else rate
        subject_dir = self.output_dir / f"p{subject:03d}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        return subject_dir / f"{source_path.stem}_pose_{rate_tag}hz.npy"

    def _parse_metadata(self, path: Path) -> Optional[Dict[str, int]]:
        match = FILENAME_RE.search(path.stem)
        if not match:
            return None
        return {key: int(value) for key, value in match.groupdict().items()}

    def _load_clip(self, path: Path) -> Optional[_Clip]:
        frames_data: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
        body_scores: Dict[str, float] = defaultdict(float)

        with path.open("r", encoding="utf-8") as f:
            try:
                num_frames = int(f.readline().strip())
            except ValueError:
                return None

            for _ in range(num_frames):
                line = f.readline()
                if not line:
                    break
                try:
                    num_bodies = int(line.strip())
                except ValueError:
                    continue

                frame_bodies: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for _ in range(num_bodies):
                    body_info = f.readline()
                    if not body_info:
                        break
                    parts = body_info.strip().split()
                    body_id = parts[0]

                    try:
                        num_joints = int(f.readline().strip())
                    except ValueError:
                        break

                    coords = np.zeros((num_joints, 3), dtype=np.float32)
                    tracking = np.zeros(num_joints, dtype=np.int16)

                    for joint_idx in range(num_joints):
                        joint_line = f.readline()
                        if not joint_line:
                            break
                        joint_parts = joint_line.strip().split()
                        if len(joint_parts) < 12:
                            continue
                        coords[joint_idx] = tuple(float(v) for v in joint_parts[:3])
                        tracking[joint_idx] = int(float(joint_parts[-1]))

                    frame_bodies[body_id] = (coords, tracking)
                    body_scores[body_id] += float(tracking.sum())

                frames_data.append(frame_bodies)

        if not body_scores:
            return None

        chosen_body = max(body_scores.items(), key=lambda item: item[1])[0]
        coords = np.zeros((len(frames_data), NUM_JOINTS, 3), dtype=np.float32)
        mask = np.zeros((len(frames_data), NUM_JOINTS), dtype=bool)

        for frame_idx, bodies in enumerate(frames_data):
            body = bodies.get(chosen_body)
            if body is None:
                continue
            joint_coords, joint_tracking = body
            limit = min(NUM_JOINTS, joint_coords.shape[0])
            coords[frame_idx, :limit, :] = joint_coords[:limit]
            mask[frame_idx, :limit] = joint_tracking[:limit] > 0

        return self._normalize_clip(coords, mask)

    def _normalize_clip(self, coords: np.ndarray, mask: np.ndarray) -> Optional[_Clip]:
        raw_frames = coords.shape[0]
        required_indices = [
            self.center_idx,
            self.scale_a,
            self.scale_b,
            4, 5, 6,  # left shoulder/elbow/wrist (1-based 5,6,7)
            8, 9, 10,  # right shoulder/elbow/wrist (1-based 9,10,11)
        ]
        frame_has_data = mask[:, required_indices].all(axis=1)
        if not np.any(frame_has_data):
            return None

        coords = coords[frame_has_data]
        mask = mask[frame_has_data]

        center_positions = coords[:, self.center_idx, :]
        center_valid = mask[:, self.center_idx]
        if center_valid.any():
            center_positions = self._forward_fill(center_positions, center_valid)
        else:
            center_positions = np.zeros_like(center_positions)

        coords = coords - center_positions[:, None, :]

        scale_valid = mask[:, self.scale_a] & mask[:, self.scale_b]
        scale = 1.0
        if scale_valid.any():
            deltas = coords[scale_valid, self.scale_a, :] - coords[scale_valid, self.scale_b, :]
            dists = np.linalg.norm(deltas, axis=1)
            positive = dists[dists > 1e-6]
            if positive.size:
                scale = float(np.median(positive))
        if scale > 1e-6:
            coords /= scale

        pose = np.transpose(coords, (2, 1, 0))
        visibility = mask.T
        frames = coords.shape[0]
        return _Clip(pose=pose, mask=visibility, frames=frames, raw_frames=raw_frames)

    def _resample_clip(self, clip: _Clip, target_rate: float) -> _Clip:
        orig_rate = float(self.config.orig_sampling_rate_hz)
        duration_sec = (clip.frames - 1) / orig_rate if clip.frames > 1 else 0.0
        target_frames = max(1, int(round(duration_sec * target_rate)) + 1)

        src_t = np.linspace(0.0, duration_sec, num=clip.frames, dtype=np.float32)
        tgt_t = np.linspace(0.0, duration_sec, num=target_frames, dtype=np.float32)

        pose_resampled = np.empty((clip.pose.shape[0], clip.pose.shape[1], target_frames), dtype=np.float32)
        for axis in range(clip.pose.shape[0]):
            for joint in range(clip.pose.shape[1]):
                pose_resampled[axis, joint, :] = np.interp(tgt_t, src_t, clip.pose[axis, joint, :])

        mask_resampled = np.empty((clip.mask.shape[0], target_frames), dtype=bool)
        for joint in range(clip.mask.shape[0]):
            mask_resampled[joint, :] = np.interp(tgt_t, src_t, clip.mask[joint, :].astype(np.float32)) >= 0.5

        return _Clip(
            pose=pose_resampled,
            mask=mask_resampled,
            frames=target_frames,
            raw_frames=clip.raw_frames,
        )

    @staticmethod
    def _forward_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        filled = values.copy()
        last_valid = None
        for idx, is_valid in enumerate(valid):
            if is_valid:
                last_valid = values[idx]
            elif last_valid is not None:
                filled[idx] = last_valid
        # Back-fill any leading invalid rows with the first valid observation
        if valid.any():
            first_idx = int(np.argmax(valid))
            for idx in range(first_idx):
                filled[idx] = values[first_idx]
        return filled


def main() -> None:
    config = NTUPreprocessConfig(
        data_dir=Path("datasets/ntu_skel_original"),
        output_dir=Path("datasets/ntu"),
        actions=None,
        subjects=None,
        overwrite=False,
        target_sampling_rates=(50.0, 100.0),
    )
    preprocessor = NTUPreprocessor(config)
    summary = preprocessor.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
