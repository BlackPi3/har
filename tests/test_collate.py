import torch
from src.datasets.common.collate import mask_collate


def test_mask_collate_variable_lengths():
    # Create three samples with different T
    J = 3
    T1, T2, T3 = 10, 7, 15
    pose1 = torch.randn(3, J, T1)
    acc1 = torch.randn(3, T1)
    pose2 = torch.randn(3, J, T2)
    acc2 = torch.randn(3, T2)
    pose3 = torch.randn(3, J, T3)
    acc3 = torch.randn(3, T3)

    batch = [
        (pose1, acc1, torch.tensor(0)),  # implicit full-length mask
        (pose2, acc2, torch.tensor(1), T2),  # explicit length
        (pose3, acc3, torch.tensor(2), torch.ones(T3, dtype=torch.bool)),  # explicit mask
    ]

    pose_pad, acc_pad, labels, mask = mask_collate(batch)

    assert pose_pad.shape == (3, 3, J, max(T1, T2, T3))
    assert acc_pad.shape == (3, 3, max(T1, T2, T3))
    assert labels.shape == (3,)
    assert mask.shape == (3, max(T1, T2, T3))

    # First sample mask should have T1 True then False
    assert mask[0].sum().item() == T1
    # Second sample mask from length
    assert mask[1].sum().item() == T2
    # Third sample provided full True mask
    assert mask[2].all()
