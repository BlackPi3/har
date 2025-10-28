"""Unified Experiment Runner (Hydra-based)

Usage examples:
    python experiments/run.py data=mmfit_debug trainer.epochs=2
    python experiments/run.py scenario=scenario2 trainer.epochs=5 optim.lr=5e-4

NOTE: For plain `python experiments/run.py ...` to work, the project
root (the directory containing `src/`) must be on PYTHONPATH. This is naturally
true if:
    1. You've run `pip install -e .` inside the active environment, OR
    2. You invoke via module form: `python -m experiments.run ...`

If you see `ModuleNotFoundError: No module named 'src'`, ensure the editable
install succeeded (activate env, then `pip install -e .`).
"""
import os
import json
from pathlib import Path
from typing import Any
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import set_seed  # type: ignore
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
    dataset_name = cfg.data.dataset_name
    data_dir = Path(cfg.env.data_dir) / dataset_name

    if not os.path.exists(data_dir):
        print(
            f"Warning: data_dir {data_dir} does not exist; proceeding anyway.")
    else:
        print(f"✓ Data directory: {data_dir}")

    print("Running Experiment")
    # Support new 'scenario' group and keep backward compatibility with 'experiment'
    exp_name = None
    if hasattr(cfg, 'scenario'):
        exp_name = getattr(cfg.scenario, 'experiment_name', None) or getattr(cfg.scenario, 'name', None)
    if exp_name is None and hasattr(cfg, 'experiment'):
        exp_name = getattr(cfg.experiment, 'experiment_name', None) or getattr(cfg.experiment, 'name', None)
    exp_name = exp_name or 'unknown_experiment'
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

    def _fallback(*candidates, default=None):
        for cand in candidates:
            if cand is None:
                continue
            if isinstance(cand, str):
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
    ns_dict["sensor_window_length"] = _fallback("data.sensor_window_length", default=None)
    try:
        if hasattr(cfg, 'scenario') and hasattr(cfg.scenario, 'alpha'):
            ns_dict["scenario2_alpha"] = cfg.scenario.alpha
        if hasattr(cfg, 'scenario') and hasattr(cfg.scenario, 'beta'):
            ns_dict["scenario2_beta"] = cfg.scenario.beta
    except Exception:
        pass
    try:
        if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'alpha'):
            ns_dict["scenario2_alpha"] = cfg.experiment.alpha
        if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'beta'):
            ns_dict["scenario2_beta"] = cfg.experiment.beta
    except Exception:
        pass
    ns_dict["dataset_name"] = dataset_name
    ns = SimpleNamespace(**ns_dict)
    ns.device = device_str
    ns.torch_device = torch_device
    ns.cluster = False

    dls = get_dataloaders(dataset_name, ns)
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
    logger = False
    use_wandb = False
    if (getattr(tcfg, 'logger', None) == 'wandb') or (hasattr(tcfg, 'wandb') and getattr(tcfg.wandb, 'enabled', False)):
        use_wandb = True
    if use_wandb:
        try:
            import os as _os
            from pytorch_lightning.loggers import WandbLogger  # type: ignore
            if hasattr(tcfg, 'wandb') and getattr(tcfg.wandb, 'mode', None) in ('offline', 'disabled'):
                _os.environ['WANDB_MODE'] = tcfg.wandb.mode
            wargs = {}
            for key in ('project', 'entity', 'group', 'name', 'tags'):
                if hasattr(tcfg, 'wandb') and getattr(tcfg.wandb, key, None):
                    wargs[key] = getattr(tcfg.wandb, key)
            if 'name' not in wargs or not wargs['name']:
                wargs['name'] = exp_name
            logger = WandbLogger(**wargs, log_model=False)
            logger.experiment.config.update({
                'alpha': getattr(cfg.experiment, 'alpha', None),
                'beta': getattr(cfg.experiment, 'beta', None),
                'lr': ns.lr,
                'weight_decay': ns.weight_decay,
                'epochs': ns.epochs,
                'dataset': dataset_name,
            }, allow_val_change=True)
            print("✓ W&B logger initialized")
        except ImportError:
            print("W&B requested but not installed. Install via 'pip install wandb'. Proceeding without logger.")
            logger = False
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

    def _to_float(x):
        try:
            if x is None:
                return None
            if hasattr(x, 'item'):
                return float(x.item())
            return float(x)
        except Exception:
            return None

    callback_metrics = getattr(pl_trainer, 'callback_metrics', {}) or {}
    last_val_loss = _to_float(callback_metrics.get('val_loss'))
    last_val_f1 = _to_float(callback_metrics.get('val_f1'))
    best_val_f1 = _to_float(getattr(lightning_module, 'best_val_f1', None))

    best_val_loss = None
    for cb in callbacks:
        cls_name = cb.__class__.__name__
        if cls_name == 'ModelCheckpoint' and getattr(cb, 'monitor', None) == tcfg.checkpoint.monitor:
            score = getattr(cb, 'best_model_score', None)
            best_val_loss = _to_float(score)
            break

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "history": history,
        "final_metrics": {
            "train_loss_last": history["train_loss"][-1] if history["train_loss"] else None,
            "val_loss_last": last_val_loss,
            "val_f1_last": last_val_f1,
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
        },
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    if use_wandb and logger:
        try:
            logger.experiment.log({
                'final/val_loss_last': last_val_loss,
                'final/val_f1_last': last_val_f1,
                'final/best_val_f1': best_val_f1,
                'final/best_val_loss': best_val_loss,
            })
        except Exception as e:
            print(f"Warning: failed to log final metrics to W&B: {e}")

    metric = None
    mode = None
    try:
        metric = getattr(cfg.hpo, 'metric', None)
        mode = getattr(cfg.hpo, 'mode', None)
    except Exception:
        metric = None
        mode = None
    metric = metric or 'val_f1'
    mode = mode or 'maximize'

    if metric == 'val_f1':
        obj = last_val_f1 if last_val_f1 is not None else best_val_f1
    elif metric == 'val_loss':
        obj = last_val_loss if last_val_loss is not None else best_val_loss
    else:
        obj = last_val_f1 if last_val_f1 is not None else best_val_f1
        metric = 'val_f1'

    if obj is None:
        obj = -1e9 if mode == 'maximize' else 1e9

    if metric == 'val_loss' and mode == 'maximize' and obj is not None:
        obj = -float(obj)
    if metric == 'val_f1' and mode == 'minimize' and obj is not None:
        obj = -float(obj)

    print("\nTrial completed! results.json saved. Returning objective:", float(obj))
    return float(obj)


if __name__ == "__main__":
    main()
