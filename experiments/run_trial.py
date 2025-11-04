#!/usr/bin/env python
"""Single-trial Experiment Runner (explicit YAML + simple overrides)

This runner approximates previous Hydra behavior without requiring Hydra.
It loads base config YAMLs, merges env/data/model/optim/trainer defaults,
applies simple key=value overrides (dot-paths supported), runs training, and
writes results.json into the requested run directory.

Usage:
  python -m experiments.run_trial env=remote data=mmfit_debug trainer.epochs=1 optim.lr=1e-3
"""
from __future__ import annotations
import sys
import os
import json
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import yaml
import torch

from src.config import merge_dicts
from src.data import get_dataloaders
from src.models import Regressor, FeatureExtractor, ActivityClassifier
from src.train import Trainer

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None


def _dict_to_ns(d: Dict[str, Any]) -> SimpleNamespace:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    return d


def _ns_to_dict(ns: Any) -> Any:
    if isinstance(ns, SimpleNamespace):
        return {k: _ns_to_dict(getattr(ns, k)) for k in vars(ns)}
    if isinstance(ns, dict):
        return {k: _ns_to_dict(v) for k, v in ns.items()}
    if isinstance(ns, list):
        return [_ns_to_dict(v) for v in ns]
    if isinstance(ns, tuple):
        return tuple(_ns_to_dict(v) for v in ns)
    if isinstance(ns, Path):
        return str(ns)
    try:
        import numpy as _np  # local import to avoid hard dependency at module level

        if isinstance(ns, (_np.generic,)):
            return ns.item()
    except Exception:
        pass
    try:
        import torch  # local import to avoid global dependency during serialization

        if isinstance(ns, torch.device):
            return str(ns)
    except Exception:
        pass
    return ns


def _apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for o in overrides or []:
        if not isinstance(o, str) or "=" not in o:
            continue
        k, v = o.split("=", 1)
        # Try to json-parse values for numbers/bools/null
        try:
            val = json.loads(v)
        except Exception:
            val = v
        # Map legacy scenario.* keys to top-level for Trainer compatibility
        if k == "scenario.alpha":
            cfg["alpha"] = val
        if k == "scenario.beta":
            cfg["beta"] = val
        parts = k.split(".")
        d = cfg
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = val
    return cfg


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _build_cfg(overrides: List[str]) -> SimpleNamespace:
    # Base config
    base = _load_yaml(REPO_ROOT / "conf/conf.yaml")

    # Determine env/data selections from overrides (defaults to conf/conf.yaml values via Hydra; here we default explicitly)
    env_sel = None
    data_sel = None
    for o in overrides:
        if o.startswith("env="):
            env_sel = o.split("=", 1)[1]
        elif o.startswith("data="):
            data_sel = o.split("=", 1)[1]
    if env_sel is None:
        env_sel = "local"
    if data_sel is None:
        data_sel = "mmfit"

    # Load selected groups akin to Hydra defaults
    env_cfg = _load_yaml(REPO_ROOT / f"conf/env/{env_sel}.yaml")
    data_cfg = _load_yaml(REPO_ROOT / f"conf/data/{data_sel}.yaml")
    # Minimal resolver for Hydra-style defaults in data configs (e.g., mmfit_debug -> mmfit)
    if isinstance(data_cfg, dict) and data_cfg.get("defaults"):
        base_data = {}
        for item in data_cfg.get("defaults", []) or []:
            ref = None
            if isinstance(item, str):
                ref = item
            elif isinstance(item, dict) and item:
                ref = list(item.keys())[0]
            if isinstance(ref, str):
                ref_yaml = _load_yaml(REPO_ROOT / f"conf/data/{ref}.yaml")
                base_data = merge_dicts(base_data, ref_yaml)
        # Overlay the rest of current file on top of merged base
        data_cfg = {k: v for k, v in (data_cfg or {}).items() if k != "defaults"}
        data_cfg = merge_dicts(base_data, data_cfg)
    reg_cfg = _load_yaml(REPO_ROOT / "conf/model/regressor/regressor.yaml")
    fe_cfg = _load_yaml(REPO_ROOT / "conf/model/feature_extractor/feature_extractor.yaml")
    clf_cfg = _load_yaml(REPO_ROOT / "conf/model/classifier/classifier.yaml")
    optim_cfg = _load_yaml(REPO_ROOT / "conf/optim/adam.yaml")
    trainer_cfg = _load_yaml(REPO_ROOT / "conf/trainer/lightning.yaml")

    # Compose a single dict
    cfg_dict: Dict[str, Any] = {}
    for part in (base, env_cfg, data_cfg):
        cfg_dict = merge_dicts(cfg_dict, part)
    # Nest model/optim/trainer under dedicated keys to avoid collisions
    cfg_dict["model"] = {
        "regressor": reg_cfg or {},
        "feature_extractor": fe_cfg or {},
        "classifier": clf_cfg or {},
    }
    cfg_dict["optim"] = optim_cfg or {}
    cfg_dict["trainer"] = merge_dicts(trainer_cfg or {}, cfg_dict.get("trainer", {}))

    # If dataset has a named subfolder under data_dir (e.g., mmfit), append it if not already included
    ds_name = cfg_dict.get("dataset_name")
    dd = cfg_dict.get("data_dir")
    if isinstance(ds_name, str) and isinstance(dd, str) and dd:
        # Only append if path does not already end with dataset name
        if Path(dd).name != ds_name:
            cfg_dict["data_dir"] = str((Path(dd) / ds_name))

    # Apply overrides last
    cfg_dict = _apply_overrides(cfg_dict, overrides)

    # Device/cluster auto-detect if not explicitly set
    is_cluster = bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_ON_NODE"))
    cfg_dict["cluster"] = is_cluster
    if not cfg_dict.get("device") or cfg_dict.get("device") in ("auto", None, ""):
        if torch.cuda.is_available():
            cfg_dict["device"] = "cuda"
        else:
            # Prefer MPS if available (Mac), else CPU
            mps = getattr(torch.backends, "mps", None)
            if mps and mps.is_available():
                cfg_dict["device"] = "mps"
            else:
                cfg_dict["device"] = "cpu"

    return _dict_to_ns(cfg_dict)


def _ensure_run_dir(overrides: List[str]) -> Path:
    # Honor hydra.run.dir override for compatibility with the orchestrator
    run_dir = None
    for o in overrides:
        if o.startswith("hydra.run.dir="):
            run_dir = o.split("=", 1)[1]
            break
    p = Path(run_dir) if run_dir else Path.cwd()
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(p)
    except Exception:
        pass
    return p


def _build_models(cfg) -> Dict[str, torch.nn.Module]:
    # Map config to model ctor args
    in_ch = getattr(cfg.model.regressor, "input_channels", 3)
    num_joints = getattr(cfg.model.regressor, "num_joints", 3)
    window_len = getattr(cfg, "sensor_window_length", None)
    if window_len is None and hasattr(cfg, "data"):
        window_len = getattr(cfg.data, "sensor_window_length", 300)
    if isinstance(window_len, dict):  # defensive: if not fully merged
        window_len = window_len.get("sensor_window_length", 300)

    fe_cfg = cfg.model.feature_extractor
    fe_args = dict(
        n_filters=getattr(fe_cfg, "n_filters", 9),
        filter_size=getattr(fe_cfg, "filter_size", 5),
        n_dense=getattr(fe_cfg, "n_dense", 100),
        n_channels=getattr(fe_cfg, "input_channels", 3),
        window_size=window_len,
        drop_prob=getattr(fe_cfg, "drop_prob", 0.2),
        pool_filter_size=getattr(fe_cfg, "pool_size", 2),
    )

    clf_cfg = cfg.model.classifier
    f_in = getattr(clf_cfg, "f_in", 100)
    n_classes = getattr(clf_cfg, "n_classes", 11)

    reg_cfg = cfg.model.regressor
    reg_kwargs = dict(
        joint_hidden_channels=getattr(reg_cfg, "joint_hidden_channels", None),
        joint_kernel_sizes=getattr(reg_cfg, "joint_kernel_sizes", None),
        joint_dilations=getattr(reg_cfg, "joint_dilations", None),
        joint_dropouts=getattr(reg_cfg, "joint_dropouts", None),
        temporal_hidden_channels=getattr(reg_cfg, "temporal_hidden_channels", None),
        temporal_kernel_size=getattr(reg_cfg, "temporal_kernel_size", None),
        temporal_dilation=getattr(reg_cfg, "temporal_dilation", None),
        temporal_dropout=getattr(reg_cfg, "temporal_dropout", None),
        fc_hidden=getattr(reg_cfg, "fc_hidden", None),
        fc_dropout=getattr(reg_cfg, "fc_dropout", 0.0),
        use_batch_norm=getattr(reg_cfg, "use_batch_norm", False),
    )
    reg_kwargs = {k: v for k, v in reg_kwargs.items() if v is not None}

    models = {
        "pose2imu": Regressor(in_ch=in_ch, num_joints=num_joints, window_length=window_len, **reg_kwargs).to(cfg.device),
        "fe": FeatureExtractor(**fe_args).to(cfg.device),
        "ac": ActivityClassifier(f_in=f_in, n_classes=n_classes).to(cfg.device),
    }
    return models


def _build_optim(cfg, models):
    params = []
    for m in models.values():
        params.extend(list(m.parameters()))
    lr = float(getattr(cfg.optim, "lr", 1e-3))
    weight_decay = float(getattr(cfg.optim, "weight_decay", 0.0))
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    sched_cfg = getattr(cfg.optim, "scheduler", {}) or {}
    name = (getattr(sched_cfg, "name", "") if not isinstance(sched_cfg, dict) else sched_cfg.get("name", ""))
    mode = (getattr(sched_cfg, "mode", "min") if not isinstance(sched_cfg, dict) else sched_cfg.get("mode", "min"))
    factor = (getattr(sched_cfg, "factor", 0.1) if not isinstance(sched_cfg, dict) else sched_cfg.get("factor", 0.1))
    patience = (getattr(sched_cfg, "patience", 10) if not isinstance(sched_cfg, dict) else sched_cfg.get("patience", 10))
    if (name or "").lower().startswith("reduce"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(factor),
            patience=int(patience),
        )
    else:
        scheduler = None
    return optimizer, scheduler


def _clean_history(history: Dict[str, List[Any]]) -> Dict[str, List[float]]:
    clean: Dict[str, List[float]] = {}
    for key, values in (history or {}).items():
        if not isinstance(values, list):
            continue
        clean[key] = [float(v) for v in values]
    return clean


def _write_results(run_dir: Path, history: Dict[str, List[Any]], best_epoch: int | None) -> None:
    history = history or {}
    val_f1_hist = history.get("val_f1", []) or []
    val_loss_hist = history.get("val_loss", []) or []

    final = {
        "val_f1_last": float(val_f1_hist[-1]) if val_f1_hist else None,
        "best_val_f1": float(max(val_f1_hist)) if val_f1_hist else None,
        "val_loss_last": float(val_loss_hist[-1]) if val_loss_hist else None,
        "best_val_loss": float(min(val_loss_hist)) if val_loss_hist else None,
    }

    out = {
        "final_metrics": final,
        "best_epoch": (int(best_epoch) + 1) if isinstance(best_epoch, int) else None,
        "history": _clean_history(history),
    }
    (run_dir / "results.json").write_text(json.dumps(out, indent=2))


def _write_history_csv(run_dir: Path, history: Dict[str, List[Any]]) -> None:
    history = history or {}
    keys = [k for k, v in history.items() if isinstance(v, list) and v]
    if not keys:
        return
    length = max(len(history[k]) for k in keys)
    csv_path = run_dir / "history.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for idx in range(length):
            row = [idx + 1]
            for key in keys:
                values = history.get(key, [])
                row.append(float(values[idx]) if idx < len(values) else "")
            writer.writerow(row)


def _save_best_state(run_dir: Path, best_state: Dict[str, Any] | None) -> None:
    if not best_state:
        return
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / "bundle.pth")
    for name, state in best_state.items():
        torch.save(state, ckpt_dir / f"{name}_best.pth")


def _save_plots(run_dir: Path, history: Dict[str, List[Any]]) -> None:
    if plt is None:
        return
    history = history or {}
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if not epochs:
        return
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _plot_pair(y1, y2, labels, title, filename):
        if not y1 and not y2:
            return
        plt.figure()
        if y1:
            plt.plot(list(epochs), y1, label=labels[0])
        if y2:
            plt.plot(list(epochs), y2, label=labels[1])
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        if y1 and y2:
            plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / filename)
        plt.close()

    _plot_pair(history.get("train_loss"), history.get("val_loss"), ("Train Loss", "Val Loss"), "Loss", "loss.png")
    _plot_pair(history.get("train_f1"), history.get("val_f1"), ("Train F1", "Val F1"), "Macro F1", "f1.png")
    _plot_pair(history.get("train_acc"), history.get("val_acc"), ("Train Accuracy", "Val Accuracy"), "Accuracy", "accuracy.png")
    _plot_pair(history.get("train_mse"), history.get("val_mse"), ("Train MSE", "Val MSE"), "MSE", "mse.png")
    _plot_pair(history.get("train_sim_loss"), history.get("val_sim_loss"), ("Train Sim Loss", "Val Sim Loss"), "Simulation Loss", "sim_loss.png")
    _plot_pair(history.get("train_act_loss"), history.get("val_act_loss"), ("Train Act Loss", "Val Act Loss"), "Activity Loss", "act_loss.png")


def _write_config(run_dir: Path, cfg: SimpleNamespace) -> None:
    try:
        cfg_dict = _ns_to_dict(cfg)
    except Exception:
        cfg_dict = {}
    if not cfg_dict:
        return
    with (run_dir / "resolved_config.yaml").open("w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)


def main():
    overrides = sys.argv[1:]
    cfg = _build_cfg(overrides)
    # Hoist commonly used fields expected at top-level by Trainer
    try:
        if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "patience"):
            cfg.patience = getattr(cfg.trainer, "patience", 10)
    except Exception:
        pass
    run_dir = _ensure_run_dir(overrides)

    trainer = None
    history: Dict[str, List[Any]] = {}
    try:
        dls = get_dataloaders(getattr(cfg, "dataset_name", "mmfit"), cfg)
        models = _build_models(cfg)
        optimizer, scheduler = _build_optim(cfg, models)
        epochs = getattr(cfg.trainer, "epochs", 1)
        device = cfg.device
        trainer = Trainer(models=models, dataloaders=dls, optimizer=optimizer, scheduler=scheduler, cfg=cfg, device=device)
        history = trainer.fit(epochs)
    except Exception as e:
        print(f"[run_trial] ERROR: {e}")
        history = {"val_f1": [], "val_loss": []}

    best_state = getattr(trainer, "best_state", None) if trainer else None
    best_epoch = getattr(trainer, "best_epoch", None) if trainer else None

    _write_results(run_dir, history, best_epoch)
    _write_history_csv(run_dir, history)
    _save_best_state(run_dir, best_state)
    _save_plots(run_dir, history)
    _write_config(run_dir, cfg)


if __name__ == "__main__":
    main()
