from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from ..common.base_dataset import BaseHARDataset


FILENAME_RE = re.compile(r"A(?P<action>\d{3})", re.IGNORECASE)


@dataclass
class _Clip:
    pose: torch.Tensor  # (3, joints, T)
    label: int
    length: int


class NTUDataset(BaseHARDataset):
    """
    Sliding-window dataset for preprocessed NTU pose clips.

    Expected inputs:
    - Pose npy files saved by the NTU preprocessor as `(3, joints, frames)` and
      stored under `datasets/ntu/p###/*_pose_{rate}hz.npy`.

    Output windows:
    - pose: (3, joints, window_length)
    - label: integer class id inferred from the filename action code (A###).
    """

    def __init__(
        self,
        data_dir: Path,
        subjects: List[str],
        sampling_rate_hz: int,
        window_length: int,
        stride_seconds: float,
        pad_short_clips: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.window_length = int(window_length)
        stride = int(round(stride_seconds * sampling_rate_hz))
        self.stride = max(1, stride)
        self.dtype = dtype

        self.ACTIONS = self._discover_actions(data_dir, sampling_rate_hz)
        self.action_to_idx = {act: idx for idx, act in enumerate(sorted(self.ACTIONS))}

        self.clips: List[_Clip] = []
        self.indices: List[Tuple[int, int]] = []

        for subject in subjects:
            subj_dir = data_dir / f"p{subject}"
            if not subj_dir.is_dir():
                continue
            for pose_path in sorted(subj_dir.glob(f"*pose_{sampling_rate_hz}hz.npy")):
                action = self._parse_action(pose_path)
                if action is None:
                    continue
                label = self.action_to_idx.get(action)
                if label is None:
                    continue
                pose_np = np.load(pose_path, mmap_mode="r")
                pose_tensor = self._prepare_clip(pose_np, pad_short_clips)
                if pose_tensor is None:
                    continue
                self._register_clip(pose_tensor, label)

    def _discover_actions(self, data_dir: Path, rate: int) -> List[str]:
        actions = set()
        for path in data_dir.glob(f"p*/**/*pose_{rate}hz.npy"):
            code = self._parse_action(path)
            if code:
                actions.add(code)
        return sorted(actions)

    def _parse_action(self, path: Path) -> str | None:
        match = FILENAME_RE.search(path.name)
        if match:
            return f"A{match.group('action')}"
        return None

    def _prepare_clip(self, pose: np.ndarray, pad_short: bool) -> torch.Tensor | None:
        if pose.ndim != 3:
            return None
        # Normalize to (3, joints, frames)
        if pose.shape[0] == 3 and pose.shape[2] < pose.shape[1]:
            pose = np.transpose(pose, (0, 2, 1))
        elif pose.shape[-1] == 3:
            pose = np.transpose(pose, (2, 1, 0))
        pose = np.array(pose, dtype=np.float32, copy=True)

        length = pose.shape[2]
        if length < self.window_length and pad_short:
            pad = self.window_length - length
            pose = np.pad(pose, ((0, 0), (0, 0), (0, pad)), mode="edge")
            length = self.window_length
        elif length < self.window_length:
            return None

        return torch.from_numpy(pose).to(dtype=self.dtype)

    def _register_clip(self, pose: torch.Tensor, label: int) -> None:
        length = pose.shape[-1]
        clip_idx = len(self.clips)
        start = 0
        added = False
        while start + self.window_length <= length:
            self.indices.append((clip_idx, start))
            added = True
            start += self.stride
        if not added:
            self.indices.append((clip_idx, 0))
        self.clips.append(_Clip(pose=pose, label=label, length=length))

    def __len__(self) -> int:
        if not self.indices:
            return 0
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        clip_idx, start = self.indices[idx]
        clip = self.clips[clip_idx]
        end = start + self.window_length
        pose = clip.pose[:, :, start:end]
        return pose, clip.label
