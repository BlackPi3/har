from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from ..common.base_dataset import BaseHARDataset


@dataclass
class _Clip:
    pose: torch.Tensor  # (3, joints, T)
    acc: torch.Tensor   # (3, T)
    label: int
    length: int


class UTDMHADDataset(BaseHARDataset):
    """Dataset that returns sliding windows from aligned UTD-MHAD clips."""

    def __init__(
        self,
        data_dir: Path,
        subjects: List[str],
        pose_suffix: str,
        acc_suffix: str,
        sensor_window_length: int,
        stride_seconds: float,
        sampling_rate_hz: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.window_length = int(sensor_window_length)
        stride = int(round(stride_seconds * sampling_rate_hz))
        self.stride = max(1, stride)
        self.dtype = dtype

        self.clips: List[_Clip] = []
        self.indices: List[Tuple[int, int]] = []
        self.ACTIONS = self._discover_actions(data_dir)
        self.action_to_idx = {act: idx for idx, act in enumerate(sorted(self.ACTIONS))}

        selected_joints = [8, 9, 10]  # right shoulder, elbow, wrist (0-based)

        for subject in subjects:
            subj_dir = data_dir / subject
            if not subj_dir.is_dir():
                continue
            for pose_path in sorted(subj_dir.glob(f"a*_{subject}_t*_{pose_suffix}")):
                prefix = pose_path.name.replace(f"_{pose_suffix}", "")
                acc_path = subj_dir / f"{prefix}_{acc_suffix}"
                if not acc_path.exists():
                    continue
                action = prefix.split("_")[0]
                label = self.action_to_idx.get(action)
                if label is None:
                    continue
                pose_np = np.load(pose_path)
                acc_np = np.load(acc_path)
                pose_tensor, acc_tensor = self._prepare_clip(pose_np, acc_np, selected_joints)
                self._register_clip(pose_tensor, acc_tensor, label)

    def _discover_actions(self, data_dir: Path) -> List[str]:
        actions = set()
        for subject_dir in data_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            for path in subject_dir.glob("a*_s*_t*_skeleton_aligned.npy"):
                actions.add(path.name.split("_", 1)[0])
        return sorted(actions)

    def _prepare_clip(self, pose: np.ndarray, acc: np.ndarray, joint_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        pose_tensor = torch.from_numpy(pose[:, joint_indices, :]).to(dtype=self.dtype)
        acc_tensor = torch.from_numpy(acc).to(dtype=self.dtype)
        return pose_tensor, acc_tensor

    def _register_clip(self, pose: torch.Tensor, acc: torch.Tensor, label: int) -> None:
        length = pose.shape[-1]
        if length < self.window_length:
            pad = self.window_length - length
            pose = torch.nn.functional.pad(pose, (0, pad))
            acc = torch.nn.functional.pad(acc, (0, pad))
            length = self.window_length
        clip_idx = len(self.clips)
        start = 0
        added = False
        while start + self.window_length <= length:
            self.indices.append((clip_idx, start))
            added = True
            start += self.stride
        if not added:
            self.indices.append((clip_idx, 0))
        self.clips.append(_Clip(pose=pose, acc=acc, label=label, length=length))

    def __len__(self) -> int:
        if not self.indices:
            return 0
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        clip_idx, start = self.indices[idx]
        clip = self.clips[clip_idx]
        end = start + self.window_length
        pose = clip.pose[:, :, start:end]
        acc = clip.acc[:, start:end]
        return pose, acc, clip.label
