"""
Utility dataset + dataloader helpers for the UTD inertial clips.

This module is intentionally lightweight so that we can explore dataset
statistics from tests without wiring the full training pipeline yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class _ClipInfo:
    acc_path: Path
    action: str
    subject: str
    trial: str


class UTDInertialDataset(Dataset):
    """
    Torch dataset over the raw (non-upsampled) inertial clips from UTD.

    Parameters
    ----------
    root : str | Path
        Base directory that contains subject folders (s1, s2, ...).
    subjects : Optional[Sequence[str]]
        Optional subset of subject folder names to load (defaults to all `s*` dirs).
    acc_file_suffix : str
        File suffix to use for inertial modality, e.g. "inertial.npy" (raw) or
        "inertial_std.npy" (the normalized/upsampled variant).
    dtype : torch.dtype
        Target dtype for returned tensors.
    select_axes : Optional[Sequence[int]]
        Optional subset of axes to keep from the inertial recording (after
        transposing to channel-first). Useful if you only need the first 3 of
        the 6 recorded channels.
    """

    def __init__(
        self,
        root: str | Path,
        subjects: Optional[Sequence[str]] = None,
        acc_file_suffix: str = "inertial_std.npy",
        dtype: torch.dtype = torch.float32,
        select_axes: Optional[Sequence[int]] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"UTD root not found: {self.root}")

        if subjects is None:
            subjects = sorted(
                d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith("s")
            )
        self.subjects = list(subjects)
        suffix = acc_file_suffix if acc_file_suffix.endswith(".npy") else f"{acc_file_suffix}.npy"
        self.acc_file_suffix = suffix
        self.dtype = dtype
        self.select_axes = tuple(select_axes) if select_axes is not None else None

        self._clips: List[_ClipInfo] = self._collect_clips()
        self.action_to_idx = self._build_action_index(self._clips)
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}

        if not self._clips:
            raise RuntimeError(f"No inertial clips found in {self.root} for suffix '{suffix}'")

    def _collect_clips(self) -> List[_ClipInfo]:
        clips: List[_ClipInfo] = []
        suffix = f"_{self.acc_file_suffix}"
        for subject in self.subjects:
            subject_dir = self.root / subject
            if not subject_dir.is_dir():
                continue
            pattern = f"a*_{subject}_t*{suffix}"
            for acc_path in sorted(subject_dir.glob(pattern)):
                prefix = acc_path.name[: -len(suffix)]
                parts = prefix.split("_")
                if len(parts) != 3:
                    continue
                action, _, trial = parts
                clips.append(_ClipInfo(acc_path=acc_path, action=action, subject=subject, trial=trial))
        return clips

    @staticmethod
    def _build_action_index(clips: Iterable[_ClipInfo]) -> Dict[str, int]:
        actions = sorted({clip.action for clip in clips})
        return {action: idx for idx, action in enumerate(actions)}

    def __len__(self) -> int:
        return len(self._clips)

    def _load_inertial(self, path: Path) -> torch.Tensor:
        data = np.load(path)
        if data.ndim != 2:
            raise ValueError(f"Unexpected inertial array shape {data.shape} for {path}")

        # Preprocessed inertial clips are stored channel-first (3, N) after dropping rotational axes.
        # Our experiments ignore the rotational channels, so callers can pass select_axes=(0,1,2)
        # to drop them without touching the raw arrays.
        # Files can be (channels, time) or (time, channels). Convert to channels-first.
        if data.shape[0] > data.shape[1]:
            data = data.T
        if self.select_axes is not None:
            data = data[self.select_axes, :]
        tensor = torch.as_tensor(data, dtype=self.dtype)
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, object]]:
        clip = self._clips[idx]
        acc = self._load_inertial(clip.acc_path)
        label = self.action_to_idx[clip.action]
        metadata = {
            "path": str(clip.acc_path),
            "subject": clip.subject,
            "action": clip.action,
            "trial": clip.trial,
            "length": acc.shape[1],
        }
        return acc, label, metadata


def make_utd_inertial_dataloader(
    root: str | Path,
    *,
    subjects: Optional[Sequence[str]] = None,
    acc_file_suffix: str = "inertial_std.npy",
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: torch.dtype = torch.float32,
    select_axes: Optional[Sequence[int]] = None,
) -> DataLoader:
    """
    Convenience wrapper that instantiates UTDInertialDataset and a torch DataLoader.
    """
    dataset = UTDInertialDataset(
        root=root,
        subjects=subjects,
        acc_file_suffix=acc_file_suffix,
        dtype=dtype,
        select_axes=select_axes,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
