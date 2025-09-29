"""Unified Experiment Runner (Hydra-based)

Use: python experiments/run_experiment.py experiment=scenario2
Override any value: python experiments/run_experiment.py epochs=5 lr=5e-4 model.regressor.sequence_length=256
"""
import os
import json
from pathlib import Path
from typing import Any
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# --- Robust import of project modules ---------------------------------------
# If user forgot to run `pip install -e .`, ensure repo root / src is on sys.path.
try:
    from src.config import set_seed  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from src.config import set_seed  # retry

from src.data import get_dataloaders
from src.models import Regressor, FeatureExtractor, ActivityClassifier
from src.lightning_module import HARLightningModule

import pytorch_lightning as pl


def _select_device(pref: str) -> str:
    if pref not in (None, "", "auto"):
        return pref
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"
    return "cpu"


def build_models(cfg, device):
    reg = cfg.model.regressor
    feat = cfg.model.feature_extractor
    clf = cfg.model.classifier
    models = {
        "pose2imu": Regressor(
            in_ch=reg.input_channels,
            num_joints=reg.num_joints,
            window_length=reg.sequence_length,
        ).to(device),
        "fe": FeatureExtractor(
            n_filters=feat.n_filters,
            filter_size=feat.filter_size,
            n_dense=feat.n_dense,
            n_channels=feat.input_channels,
            window_size=feat.window_size,
            drop_prob=feat.drop_prob,
            pool_filter_size=feat.pool_size,
        ).to(device),
        "ac": ActivityClassifier(f_in=clf.f_in, n_classes=clf.n_classes).to(device),
    }
    return models


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@hydra.main(config_path="../conf", config_name="conf", version_base=None)
def main(cfg: DictConfig) -> Any:
    set_seed(cfg.seed)
    device_str = _select_device(getattr(cfg, "device", "auto"))
    torch_device = torch.device(device_str)

    # Data directory resolution (simplified): rely on explicit config.
    data_name = cfg.data.data_name
    data_dir = Path(cfg.env.data_dir) / data_name

    if not os.path.exists(data_dir):
        print(
            f"Warning: data_dir {data_dir} does not exist; proceeding anyway.")
    else:
        print(f"✓ Data directory: {data_dir}")

    print("Running Experiment")
    exp_name = getattr(cfg.experiment, "experiment_name", getattr(
        cfg.experiment, "name", "unknown_experiment"))
    print(f"Experiment: {exp_name}")
    print(f"Device: {device_str}")
    print(f"Dataset: {data_dir}")
    print(f"Working directory (Hydra run dir): {Path.cwd()}")

    # Build lightweight namespace expected by dataloaders (legacy trainer removed)
    from types import SimpleNamespace
    legacy_keys = [
        "data_dir",
        "pose_file",
        "acc_file",
        "sim_acc_file",
        "labels_file",
        "train_subjects",
        "val_subjects",
        "test_subjects",
    ]
    ns_dict = {k: cfg.data[k] for k in legacy_keys if k in cfg.data}
    ns_dict['data_dir'] = data_dir
    # Flexible fallback retrieval for common hyperparameters across groups

    def _fallback(*candidates, default=None):
        for cand in candidates:
            if cand is None:
                continue
            if isinstance(cand, str):
                # dotted path access
                node = cfg
                ok = True
                for part in cand.split('.'):
                    if part in node:
                        node = node[part]
                    else:
                        ok = False
                        break
                if ok:
                    return node
        return default

    ns_dict["batch_size"] = _fallback(
        "batch_size", "data.batch_size", default=64)
    ns_dict["num_workers"] = _fallback(
        "num_workers", "data.num_workers", "env.num_workers", default=4)
    ns_dict["epochs"] = _fallback("epochs", "trainer.epochs", default=100)
    ns_dict["patience"] = _fallback("patience", "trainer.patience", default=10)
    ns_dict["lr"] = float(_fallback("lr", "optim.lr", default=1e-3))
    ns_dict["weight_decay"] = float(
        _fallback("weight_decay", "optim.weight_decay", default=0.0))
    # Map experiment alpha/beta to legacy names expected by Trainer
    if "experiment" in cfg and "alpha" in cfg.experiment:
        ns_dict["scenario2_alpha"] = cfg.experiment.alpha
    if "experiment" in cfg and "beta" in cfg.experiment:
        ns_dict["scenario2_beta"] = cfg.experiment.beta
    # Ensure namespace uses the resolved dataset_name (from any accepted key)
    ns_dict["dataset_name"] = data_name
    ns = SimpleNamespace(**ns_dict)
    ns.device = device_str
    ns.torch_device = torch_device
    ns.cluster = False

    dls = get_dataloaders(data_name, ns)
    print("Dataset sizes:", {k: len(v.dataset) for k, v in dls.items()})

    models = build_models(cfg, torch_device)
    print("Parameter counts:", {k: count_params(m) for k, m in models.items()})

    if pl is None:
        raise RuntimeError("pytorch-lightning must be installed for training")
    lightning_module = HARLightningModule(cfg, ns, models)
    tcfg = cfg.trainer
    callbacks = []
    if getattr(tcfg.early_stopping, 'enabled', False):
        from pytorch_lightning.callbacks import EarlyStopping
        callbacks.append(EarlyStopping(
            monitor=tcfg.early_stopping.monitor,
            mode=tcfg.early_stopping.mode,
            patience=tcfg.early_stopping.patience,
            min_delta=tcfg.early_stopping.get('min_delta', 0.0),
            verbose=False,
        ))
    if getattr(tcfg.checkpoint, 'enabled', False):
        from pytorch_lightning.callbacks import ModelCheckpoint
        ckpt_dir = Path(tcfg.checkpoint.dirpath)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(ModelCheckpoint(
            monitor=tcfg.checkpoint.monitor,
            mode=tcfg.checkpoint.mode,
            save_top_k=tcfg.checkpoint.save_top_k,
            dirpath=str(ckpt_dir),
            filename=tcfg.checkpoint.filename,
            save_last=tcfg.checkpoint.get('save_last', True),
        ))
    logger = tcfg.logger if tcfg.logger else False
    pl_trainer = pl.Trainer(
        max_epochs=ns.epochs,
        accelerator=tcfg.accelerator,
        devices=tcfg.devices,
        precision=tcfg.precision,
        deterministic=tcfg.deterministic,
        log_every_n_steps=tcfg.log_every_n_steps,
        gradient_clip_val=tcfg.get('gradient_clip_val', None),
        enable_checkpointing=tcfg.enable_checkpointing,
        callbacks=callbacks,
        logger=logger,
    )
    print(f"\n[Lightning] Starting training for {ns.epochs} epochs...")
    pl_trainer.fit(lightning_module, train_dataloaders=dls['train'], val_dataloaders=dls['val'])
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "history": history,
        "final_metrics": {
            "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        },
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nExperiment completed! results.json saved.")
    return history


if __name__ == "__main__":  # pragma: no cover
    main()
