import torch
from types import SimpleNamespace

from src.models import Regressor, FeatureExtractor, ActivityClassifier
from src.lightning_module import HARLightningModule


def test_forward_and_loss():
    # Minimal synthetic shapes consistent with default configs
    batch_size = 2
    pose_channels = 3  # guess: adapt if different in config
    num_joints = 5
    window_len = 64
    acc_channels = 3
    n_classes = 5

    # Create dummy models with small dims (mirror expected constructor usage)
    reg = Regressor(in_ch=pose_channels, num_joints=num_joints, window_length=window_len)
    fe = FeatureExtractor(n_filters=4, filter_size=3, n_dense=16, n_channels=acc_channels, window_size=window_len, drop_prob=0.1, pool_filter_size=2)
    ac = ActivityClassifier(f_in=16, n_classes=n_classes)

    cfg = SimpleNamespace(experiment=SimpleNamespace(alpha=1.0, beta=0.0))
    ns = SimpleNamespace(lr=1e-3, weight_decay=0.0, patience=3)
    models = {"pose2imu": reg, "fe": fe, "ac": ac}

    module = HARLightningModule(cfg, ns, models)

    # Regressor expects input shape (B, C, J, T) because it uses Conv2d over joints x time
    pose = torch.randn(batch_size, pose_channels, num_joints, window_len)
    # Sanity checks to make failure modes clearer
    assert pose.shape == (batch_size, pose_channels, num_joints, window_len)
    acc = torch.randn(batch_size, acc_channels, window_len)
    labels = torch.randint(0, n_classes, (batch_size,))

    total, mse_l, sim_l, act_l, logits, labels_out = module._shared_step((pose, acc, labels))
    assert total.requires_grad
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == n_classes
    assert labels_out.shape[0] == batch_size
