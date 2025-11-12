"""
Batch collation utilities for HAR datasets.

Provides a mask-aware collate_fn that pads variable-length temporal windows
to the maximum length in the batch and returns a boolean mask indicating
valid timesteps. This enables combining datasets with different windowing
policies (e.g., continuous sliding windows vs. clip-based variable durations)
without changing model code.

Expected batch item structure:
    (pose, acc, label[, length_or_mask])

Where:
    - pose: Tensor[3, J, T]   (channels x joints x time)
    - acc:  Tensor[3, T]      (channels x time)
    - label: Tensor[] or int
    - optional length_or_mask: either an int length (T_valid) or a boolean
      mask Tensor[T] with True for valid positions. If not provided, the
      function assumes the full T is valid for each item.

Outputs:
    - pose: Tensor[B, 3, J, T_max]
    - acc:  Tensor[B, 3, T_max]
    - labels: Tensor[B]
    - mask: Tensor[B, T_max] (bool)

This collate is a drop-in replacement for torch.utils.data.DataLoader(collate_fn=...).
Use it when batching variable-length sequences (e.g., UTD-MHAD clips) or when you
want explicit masks for attention/temporal pooling.
"""
from __future__ import annotations
from typing import List, Tuple, Union, Any
import torch


BatchItem = Tuple[torch.Tensor, torch.Tensor, Union[int, torch.Tensor], Any]


def _infer_length_or_mask(item: BatchItem):
    pose, acc, label, *rest = item
    T = pose.shape[-1]
    if rest:
        lm = rest[0]
        if isinstance(lm, int):
            length = max(0, min(int(lm), T))
            mask = torch.zeros(T, dtype=torch.bool)
            if length > 0:
                mask[:length] = True
            return length, mask
        if torch.is_tensor(lm):
            lm = lm.to(dtype=torch.bool)
            if lm.numel() != T:
                raise ValueError(f"Provided mask length {lm.numel()} != window length {T}")
            length = int(lm.long().sum().item())
            return length, lm
    # default: everything valid
    mask = torch.ones(T, dtype=torch.bool)
    return T, mask


def pad_and_stack(batch: List[BatchItem]):
    # Collect dims and determine T_max
    poses, accs, labels, lengths, masks = [], [], [], [], []
    T_max = 0
    J = None
    for item in batch:
        pose, acc, label, *rest = item
        if pose.ndim != 3 or acc.ndim != 2:
            raise ValueError(f"Unexpected shapes in batch item: pose {pose.shape}, acc {acc.shape}")
        if J is None:
            J = pose.shape[1]
        T = pose.shape[-1]
        T_max = max(T_max, T)
        length, mask = _infer_length_or_mask(item)
        poses.append(pose)
        accs.append(acc)
        labels.append(label if torch.is_tensor(label) else torch.tensor(label, dtype=torch.long))
        lengths.append(length)
        masks.append(mask)

    # Allocate padded tensors
    B = len(batch)
    device = poses[0].device
    dtype_pose = poses[0].dtype
    dtype_acc = accs[0].dtype

    pose_pad = torch.zeros((B, 3, J, T_max), dtype=dtype_pose, device=device)
    acc_pad = torch.zeros((B, 3, T_max), dtype=dtype_acc, device=device)
    mask_pad = torch.zeros((B, T_max), dtype=torch.bool, device=device)
    label_t = torch.stack([lbl.to(device=device) for lbl in labels], dim=0)

    for i, (p, a, m) in enumerate(zip(poses, accs, masks)):
        T = p.shape[-1]
        pose_pad[i, :, :, :T] = p
        acc_pad[i, :, :T] = a
        mask_pad[i, :T] = m.to(device=device)

    return pose_pad, acc_pad, label_t, mask_pad


def mask_collate(batch: List[BatchItem]):
    """Mask-aware collate function.

    This is safe for both fixed- and variable-length batches. If all windows
    have equal T, it behaves like a simple stack and produces an all-True mask.
    """
    return pad_and_stack(batch)
