#!/usr/bin/env python
"""Single-trial Experiment Runner (explicit YAML + simple overrides)

This runner approximates previous Hydra behavior without requiring Hydra.
It loads base config YAMLs, merges env/data/model/optim/trainer defaults,
applies simple key=value overrides (dot-paths supported), runs training, and
writes results.json into the requested run directory.

Usage:
  python -m experiments.run_trial trial=scenario2_mmfit env=remote data=mmfit_debug trainer.epochs=1 optim.lr=1e-3
"""
from __future__ import annotations
import sys
import os
import json
import math
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

import yaml
import torch

from src.config import merge_dicts, get_data_cfg_value, set_seed
from src.data import get_dataloaders
from src.models import Regressor, FeatureExtractor, ActivityClassifier
from src.train_scenario2 import Trainer

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None

_SKIP_ARTIFACTS = str(
    os.environ.get(
        "RUN_TRIAL_SKIP_ARTIFACTS",
        os.environ.get("RUN_TRIAL_SKIP_CHECKPOINTS", ""),
    )
).lower() in ("1", "true", "yes")


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
    alias_map = {
        "scenario.alpha": "alpha",
        "scenario.beta": "beta",
        "scenario.gamma": "gamma",
        "trial.alpha": "alpha",
        "trial.beta": "beta",
        "trial.gamma": "gamma",
    }
    for o in overrides or []:
        if not isinstance(o, str) or "=" not in o:
            continue
        k, v = o.split("=", 1)
        # Try to json-parse values for numbers/bools/null
        try:
            val = json.loads(v)
        except Exception:
            val = v
        # Map scenario./trial. knobs to top-level for Trainer compatibility
        target = alias_map.get(k)
        if target:
            cfg[target] = val
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


def _load_composite_yaml(root: Path, name: str, visited: set[str] | None = None) -> Dict[str, Any]:
    """Load a YAML file that may declare simple Hydra-style defaults within the same directory."""
    visited = visited or set()
    if not name or name in visited:
        return {}
    visited.add(name)

    path = root / f"{name}.yaml"
    data = _load_yaml(path)
    if not data:
        return {}

    defaults = data.get("defaults", []) or []
    merged: Dict[str, Any] = {}
    for item in defaults:
        # Skip special _self_ marker; only simple string references are supported
        if isinstance(item, str):
            if item == "_self_":
                continue
            merged = merge_dicts(merged, _load_composite_yaml(root, item, visited))
        elif isinstance(item, dict) and item:
            # Handle dict syntax like {"trial": "base"} (or legacy {"scenario": "base"}) by consuming the value
            ref = next(iter(item.values()))
            if isinstance(ref, str):
                merged = merge_dicts(merged, _load_composite_yaml(root, ref, visited))
    data = {k: v for k, v in data.items() if k != "defaults"}
    return merge_dicts(merged, data)


def _load_data_config(name: str, visited: set[str] | None = None) -> Dict[str, Any]:
    """Load data config with simple defaults support (mirrors Hydra behavior)."""
    if not name:
        return {}
    visited = visited or set()
    if name in visited:
        return {}
    visited.add(name)
    path = REPO_ROOT / f"conf/data/{name}.yaml"
    data = _load_yaml(path)
    if not data:
        raise FileNotFoundError(f"Data config not found: {path}")
    defaults = data.get("defaults", []) or []
    merged: Dict[str, Any] = {}
    for item in defaults:
        ref = None
        if isinstance(item, str):
            ref = item
        elif isinstance(item, dict) and item:
            ref = list(item.keys())[0]
        if isinstance(ref, str):
            merged = merge_dicts(merged, _load_data_config(ref, visited))
    data = {k: v for k, v in data.items() if k != "defaults"}
    return merge_dicts(merged, data)


def _build_cfg(overrides: List[str]) -> SimpleNamespace:
    # Base config
    base = _load_yaml(REPO_ROOT / "conf/conf.yaml")
    # Prepare model defaults that can be overridden by trial config
    reg_cfg = _load_yaml(REPO_ROOT / "conf/model/regressor/regressor.yaml")
    fe_cfg = _load_yaml(REPO_ROOT / "conf/model/feature_extractor/feature_extractor.yaml")
    clf_cfg = _load_yaml(REPO_ROOT / "conf/model/classifier/classifier.yaml")
    optim_cfg = _load_yaml(REPO_ROOT / "conf/optim/adam.yaml")

    # Determine env/data selections from overrides (must be provided)
    env_sel = None
    data_sel_override = None
    trial_sel = None
    legacy_scenario_sel = None
    for o in overrides:
        if o.startswith("env="):
            env_sel = o.split("=", 1)[1]
        elif o.startswith("data="):
            data_sel_override = o.split("=", 1)[1]
        elif o.startswith("trial="):
            trial_sel = o.split("=", 1)[1]
        elif o.startswith("scenario="):
            legacy_scenario_sel = o.split("=", 1)[1]
    if env_sel is None:
        raise ValueError("env selection is required (env=<name>)")
    if trial_sel is None:
        trial_sel = legacy_scenario_sel
    if trial_sel is None:
        raise ValueError("trial selection is required (trial=<name>)")

    # Load selected groups akin to Hydra defaults
    env_cfg = _load_yaml(REPO_ROOT / f"conf/env/{env_sel}.yaml")
    trial_cfg = _load_composite_yaml(REPO_ROOT / "conf/trial", trial_sel)
    # Compose a single dict
    cfg_dict: Dict[str, Any] = {}
    for part in (base, env_cfg):
        cfg_dict = merge_dicts(cfg_dict, part)
    # Nest model/optim/trainer under dedicated keys to avoid collisions
    cfg_dict["model"] = {
        "regressor": reg_cfg or {},
        "feature_extractor": fe_cfg or {},
        "classifier": clf_cfg or {},
    }
    cfg_dict["optim"] = optim_cfg or {}

    # Merge trial-level overrides last so they remain the single source of truth per flow
    cfg_dict = merge_dicts(cfg_dict, trial_cfg)

    # Allow trial configs to declare data selection + overrides while still mutating the flat namespace.
    data_meta = cfg_dict.pop("data", None)
    data_overrides: Dict[str, Any] = {}
    data_sel = data_sel_override
    if isinstance(data_meta, dict):
        meta_selection = (
            data_meta.get("name")
            or data_meta.get("selection")
            or data_meta.get("dataset_name")
            or data_meta.get("dataset")
        )
        if meta_selection and not data_sel_override:
            data_sel = meta_selection
        overrides_node = data_meta.get("overrides")
        if isinstance(overrides_node, dict):
            data_overrides = merge_dicts(data_overrides, overrides_node)
        inline_overrides = {
            k: v
            for k, v in data_meta.items()
            if k not in ("name", "selection", "dataset", "dataset_name", "overrides")
        }
        if inline_overrides:
            data_overrides = merge_dicts(data_overrides, inline_overrides)
    elif isinstance(data_meta, str) and not data_sel_override:
        data_sel = data_meta

    if not data_sel:
        raise ValueError(
            "Dataset selection is missing. Provide data=<name> override or set data.name in the trial/HPO config."
        )

    data_cfg = _load_data_config(data_sel)
    final_data = merge_dicts(data_cfg, data_overrides or {})
    final_data["name"] = data_sel
    final_data.setdefault("dataset_name", data_sel)
    cfg_dict["data"] = final_data

    # If dataset has a named subfolder under data_dir (e.g., mmfit), append it if not already included
    ds_name = final_data.get("dataset_name")
    dd = cfg_dict.get("data_dir")
    if isinstance(ds_name, str) and isinstance(dd, str) and dd:
        # Only append if path does not already end with dataset name
        if Path(dd).name != ds_name:
            cfg_dict["data_dir"] = str((Path(dd) / ds_name))

    # Apply overrides last
    cfg_dict = _apply_overrides(cfg_dict, overrides)

    # Resolve trainer selection (required)
    trainer_node = cfg_dict.get("trainer")
    trainer_sel = None
    trainer_overrides: Dict[str, Any] = {}
    if isinstance(trainer_node, str):
        trainer_sel = trainer_node
    elif isinstance(trainer_node, dict):
        trainer_sel = trainer_node.get("name") or trainer_node.get("selection")
        trainer_overrides = {k: v for k, v in trainer_node.items() if k not in ("name", "selection")}
    if not trainer_sel:
        raise ValueError(
            "Trainer selection is missing. Specify trainer.name (or trainer=...) in the trial/config overrides."
        )
    trainer_cfg = _load_yaml(REPO_ROOT / f"conf/trainer/{trainer_sel}.yaml")
    cfg_dict["trainer"] = merge_dicts(trainer_cfg or {}, trainer_overrides or {})
    cfg_dict["trainer"]["name"] = trainer_sel

    # Stamp the selected config surface explicitly for reproducibility
    if not isinstance(cfg_dict.get("env"), dict):
        cfg_dict["env"] = env_sel
    data_field = cfg_dict.get("data")
    if isinstance(data_field, dict):
        cfg_dict["data"].setdefault("name", data_sel)
    elif data_sel:
        cfg_dict["data"] = {"name": data_sel}
    cfg_dict["trial"] = trial_sel
    # Keep legacy key for downstream consumers that still expect `scenario`
    cfg_dict["scenario"] = trial_sel

    # Drop Hydra-specific metadata so resolved configs only contain final values
    cfg_dict.pop("defaults", None)
    cfg_dict.pop("hydra", None)

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


def _ensure_run_dir(cfg: SimpleNamespace, overrides: List[str]) -> Path:
    """Determine and materialize the run directory."""
    run_dir: str | Path | None = None

    run_cfg = getattr(cfg, "run", None)
    if isinstance(run_cfg, SimpleNamespace):
        for attr in ("dir", "path", "output_dir"):
            candidate = getattr(run_cfg, attr, None)
            if candidate:
                run_dir = candidate
                break
    if run_dir is None:
        run_dir = getattr(cfg, "run_dir", None) or getattr(cfg, "output_dir", None)
    if run_dir is None:
        run_dir = os.environ.get("RUN_TRIAL_DIR") or os.environ.get("RUN_DIR")

    if run_dir is None:
        experiments_root = getattr(cfg, "experiments_dir", None)
        experiments_root = Path(experiments_root) if experiments_root else (Path.cwd() / "experiments")

        group = None
        if isinstance(run_cfg, SimpleNamespace):
            group = getattr(run_cfg, "group", None)
        if not group:
            group = os.environ.get("RUN_TRIAL_GROUP") or "best_run"

        scenario = getattr(cfg, "trial", None) or getattr(cfg, "scenario", getattr(cfg, "experiment_name", "scenario"))

        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        label = None
        if isinstance(run_cfg, SimpleNamespace):
            label = getattr(run_cfg, "label", None) or getattr(run_cfg, "name", None)
        if not label:
            label = os.environ.get("RUN_TRIAL_LABEL")
        suffix = f"{timestamp}-{label}" if label else timestamp

        if group == "hpo":
            study = getattr(run_cfg, "study", None) if isinstance(run_cfg, SimpleNamespace) else None
            if not study:
                study = getattr(cfg, "study_name", getattr(cfg, "experiment_name", scenario))
            run_dir = experiments_root / "hpo" / str(study) / suffix
        else:
            run_dir = experiments_root / str(group) / str(scenario) / suffix

    p = Path(run_dir)
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
    window_len = get_data_cfg_value(cfg, "sensor_window_length", None)
    if window_len is None:
        window_len = getattr(cfg, "sensor_window_length", None)
    if isinstance(window_len, dict):  # defensive: if not fully merged
        window_len = window_len.get("sensor_window_length", 300)

    fe_cfg = cfg.model.feature_extractor
    def _fe_args_from_cfg(node):
        return dict(
            n_filters=getattr(node, "n_filters", 9),
            filter_size=getattr(node, "filter_size", 5),
            n_dense=getattr(node, "n_dense", 100),
            n_channels=getattr(node, "input_channels", 3),
            window_size=window_len,
            drop_prob=getattr(node, "drop_prob", 0.2),
            pool_filter_size=getattr(node, "pool_size", 2),
        )
    fe_args = _fe_args_from_cfg(fe_cfg)
    fe_sim_cfg = getattr(cfg.model, "feature_extractor_sim", None)
    fe_sim_args = _fe_args_from_cfg(fe_sim_cfg) if fe_sim_cfg else fe_args

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

    trainer_cfg = getattr(cfg, "trainer", None)
    separate_cls = bool(getattr(trainer_cfg, "separate_classifiers", False)) if trainer_cfg else False
    dual_fe = bool(getattr(trainer_cfg, "separate_feature_extractors", False)) if trainer_cfg else False

    models = {
        "pose2imu": Regressor(in_ch=in_ch, num_joints=num_joints, window_length=window_len, **reg_kwargs).to(cfg.device),
        "fe": FeatureExtractor(**fe_args).to(cfg.device),
        "ac": ActivityClassifier(f_in=f_in, n_classes=n_classes).to(cfg.device),
    }
    if separate_cls:
        models["ac_sim"] = ActivityClassifier(f_in=f_in, n_classes=n_classes).to(cfg.device)
    if dual_fe:
        models["fe_sim"] = FeatureExtractor(**fe_sim_args).to(cfg.device)
    return models


def _build_optim(cfg, models):
    trainer_cfg = getattr(cfg, "trainer", None)
    modules_cfg = getattr(trainer_cfg, "trainable_modules", None) if trainer_cfg else None

    def _module_enabled(name: str) -> bool:
        if modules_cfg is None:
            return True
        value = getattr(modules_cfg, name, None)
        # Allow descriptive aliases without forcing config churn
        if value is None:
            alias_map = {
                "fe": "feature_extractor",
                "fe_sim": "feature_extractor",
                "ac": "activity_classifier",
                "ac_sim": "activity_classifier",
                "pose2imu": "regressor",
            }
            alias = alias_map.get(name)
            if alias:
                value = getattr(modules_cfg, alias, None)
        return True if value is None else bool(value)

    params = []
    frozen = []
    trainable = []
    for name, module in models.items():
        enabled = _module_enabled(name)
        module.requires_grad_(enabled)
        if enabled:
            params.extend(list(module.parameters()))
            trainable.append(name)
        else:
            frozen.append(name)
    if not params:
        raise ValueError("No trainable parameters selected; check trainer.trainable_modules settings.")
    if frozen:
        print(f"[run_trial] Freezing modules (no optimizer params): {', '.join(sorted(frozen))}")
    if trainable:
        print(f"[run_trial] Optimizing modules: {', '.join(sorted(trainable))}")

    lr = float(getattr(cfg.optim, "lr", 1e-3))
    weight_decay = float(getattr(cfg.optim, "weight_decay", 0.0))
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    warmup_epochs = int(getattr(cfg.optim, "warmup_epochs", 0) or 0)
    sched_cfg = getattr(cfg.optim, "scheduler", {}) or {}
    name = (getattr(sched_cfg, "name", "") if not isinstance(sched_cfg, dict) else sched_cfg.get("name", ""))
    mode = (getattr(sched_cfg, "mode", "min") if not isinstance(sched_cfg, dict) else sched_cfg.get("mode", "min"))
    factor = (getattr(sched_cfg, "factor", 0.1) if not isinstance(sched_cfg, dict) else sched_cfg.get("factor", 0.1))
    patience = (getattr(sched_cfg, "patience", 10) if not isinstance(sched_cfg, dict) else sched_cfg.get("patience", 10))
    if (name or "").lower().startswith("reduce"):
        if warmup_epochs > 0:
            print("[run_trial] Skipping ReduceLROnPlateau because warmup_epochs > 0")
            scheduler = None
        else:
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


def _write_results(
    run_dir: Path,
    history: Dict[str, List[Any]],
    best_epoch: int | None,
    include_history: bool,
    objective_meta: Dict[str, Any] | None = None,
    test_metrics: Dict[str, Any] | None = None,
) -> None:
    history = history or {}
    val_f1_hist = history.get("val_f1", []) or []
    val_loss_hist = history.get("val_loss", []) or []
    train_f1_hist = history.get("train_f1", []) or []
    train_loss_hist = history.get("train_loss", []) or []

    final = {
        "val_f1_last": float(val_f1_hist[-1]) if val_f1_hist else None,
        "best_val_f1": float(max(val_f1_hist)) if val_f1_hist else None,
        "val_loss_last": float(val_loss_hist[-1]) if val_loss_hist else None,
        "best_val_loss": float(min(val_loss_hist)) if val_loss_hist else None,
        "train_f1_last": float(train_f1_hist[-1]) if train_f1_hist else None,
        "train_loss_last": float(train_loss_hist[-1]) if train_loss_hist else None,
    }
    if test_metrics:
        for k, v in test_metrics.items():
            try:
                final[k] = float(v)
            except Exception:
                final[k] = v

    out: Dict[str, Any] = {
        "final_metrics": final,
        "best_epoch": (int(best_epoch) + 1) if isinstance(best_epoch, int) else None,
    }
    if include_history:
        out["history"] = _clean_history(history)
    if objective_meta:
        meta = dict(objective_meta)
        for key in ("best_score", "last_score"):
            if key in meta and meta[key] is not None:
                try:
                    meta[key] = float(meta[key])
                except Exception:
                    pass
        out["objective"] = meta
    (run_dir / "results.json").write_text(json.dumps(out, indent=2))


def _save_best_state(run_dir: Path, best_state: Dict[str, Any] | None) -> None:
    if not best_state:
        return
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / "bundle.pth")
    for name, state in best_state.items():
        torch.save(state, ckpt_dir / f"{name}_best.pth")


def _save_plots(run_dir: Path, history: Dict[str, List[Any]], best_epoch: int | None) -> None:
    if plt is None:
        return
    history = history or {}
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        return
    best_epoch_idx = int(best_epoch) if isinstance(best_epoch, int) else None
    if best_epoch_idx is not None and best_epoch_idx >= len(epochs):
        best_epoch_idx = None
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _annotate_best_epoch(ax, show_label=True):
        if best_epoch_idx is None or ax is None:
            return
        epoch_marker = best_epoch_idx + 1  # convert to 1-indexed for readability
        ax.axvline(epoch_marker, color="#666666", linestyle="--", linewidth=1, alpha=0.8)
        if not show_label:
            return
        y_min, y_max = ax.get_ylim()
        if not (math.isfinite(y_min) and math.isfinite(y_max)):
            return
        if y_max == y_min:
            text_y = y_max
        else:
            text_y = y_max - 0.05 * (y_max - y_min)
        ax.text(
            epoch_marker,
            text_y,
            f"Best Epoch {epoch_marker}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
            color="#444444",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
            clip_on=False,
        )

    def _maybe_set_log_scale(ax, values):
        if ax is None or not values:
            return False
        positive_values = [v for v in values if v is not None and v > 0]
        if not positive_values:
            return False
        ax.set_yscale("log")
        return True

    def _plot_pair(y1, y2, labels, title, filename, *, log_scale=False, use_dual_axis=False):
        if not y1 and not y2:
            return
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx() if use_dual_axis and y1 and y2 else None
        train_color = "#1f77b4"
        val_color = "#ff7f0e"

        if y1:
            ax1.plot(epochs, y1, label=labels[0], color=train_color)
            ax1.set_ylabel(labels[0] if use_dual_axis else title)
        if y2:
            target_ax = ax2 if ax2 else ax1
            target_ax.plot(epochs, y2, label=labels[1], color=val_color)
            if ax2:
                ax2.set_ylabel(labels[1])

        ax1.set_xlabel("Epoch")
        ax1.set_title(title)

        if log_scale:
            applied = _maybe_set_log_scale(ax1, y1 if y1 else y2)
            if ax2:
                _maybe_set_log_scale(ax2, y2)
            elif not applied and y2:
                _maybe_set_log_scale(ax1, y2)

        _annotate_best_epoch(ax1, show_label=True)
        if ax2:
            _annotate_best_epoch(ax2, show_label=False)

        if y1 and y2:
            if ax2:
                lines, labels_combined = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels_combined + labels2, loc="best")
            else:
                ax1.legend(loc="best")

        ax1.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / filename)
        plt.close(fig)

    _plot_pair(
        history.get("train_loss"),
        history.get("val_loss"),
        ("Train Loss", "Val Loss"),
        "Loss",
        "loss.png",
        log_scale=True,
        use_dual_axis=True,
    )
    _plot_pair(history.get("train_f1"), history.get("val_f1"), ("Train F1", "Val F1"), "Macro F1", "f1.png")
    _plot_pair(history.get("train_acc"), history.get("val_acc"), ("Train Accuracy", "Val Accuracy"), "Accuracy", "accuracy.png")
    _plot_pair(
        history.get("train_mse"),
        history.get("val_mse"),
        ("Train MSE", "Val MSE"),
        "MSE",
        "mse.png",
        use_dual_axis=True,
    )
    _plot_pair(
        history.get("train_sim_loss"),
        history.get("val_sim_loss"),
        ("Train Sim Loss", "Val Sim Loss"),
        "Simulation Loss",
        "sim_loss.png",
        use_dual_axis=True,
    )
    _plot_pair(
        history.get("train_act_loss"),
        history.get("val_act_loss"),
        ("Train Act Loss", "Val Act Loss"),
        "Activity Loss",
        "act_loss.png",
        log_scale=True,
        use_dual_axis=True,
    )


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
    try:
        set_seed(getattr(cfg, "seed", None))
    except Exception:
        pass
    if getattr(cfg, "deterministic", False):
        try:
            dev_str = str(getattr(cfg, "device", "")).lower()
            if "cuda" in dev_str or torch.cuda.is_available():
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    # Hoist commonly used fields expected at top-level by Trainer
    try:
        if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "patience"):
            cfg.patience = getattr(cfg.trainer, "patience", 10)
    except Exception:
        pass
    run_dir = _ensure_run_dir(cfg, overrides)

    trainer = None
    history: Dict[str, List[Any]] = {}
    test_metrics: Dict[str, Any] | None = None
    try:
        dataset_name = get_data_cfg_value(cfg, "dataset_name", None) or get_data_cfg_value(cfg, "name", None)
        if not dataset_name:
            dataset_name = getattr(cfg, "dataset_name", "mmfit")
        dls = get_dataloaders(dataset_name, cfg)
        models = _build_models(cfg)
        optimizer, scheduler = _build_optim(cfg, models)
        epochs = getattr(cfg.trainer, "epochs", 1)
        device = cfg.device
        trainer = Trainer(models=models, dataloaders=dls, optimizer=optimizer, scheduler=scheduler, cfg=cfg, device=device)
        history = trainer.fit(epochs)
        run_group = getattr(getattr(cfg, "run", None), "group", None)
        # Evaluate on test split (best model is restored inside fit) only for eval runs
        if run_group == "eval" and dls.get("test") is not None:
            with torch.no_grad():
                test_loss, test_f1, test_mse, test_sim, test_act, test_acc = trainer._run_epoch("test")
            test_metrics = {
                "test_loss": test_loss,
                "test_f1": test_f1,
                "test_mse": test_mse,
                "test_sim_loss": test_sim,
                "test_act_loss": test_act,
                "test_acc": test_acc,
            }
    except Exception as e:
        print(f"[run_trial] ERROR: {e}")
        history = {"val_f1": [], "val_loss": []}

    best_state = getattr(trainer, "best_state", None) if trainer else None
    best_epoch = getattr(trainer, "best_epoch", None) if trainer else None

    include_history = not _SKIP_ARTIFACTS
    objective_meta = None
    if trainer and getattr(trainer, "objective_metric", None):
        metric_name = getattr(trainer, "objective_metric", None)
        objective_meta = {
            "metric": metric_name,
            "mode": getattr(trainer, "objective_mode", None),
            "best_score": getattr(trainer, "best_score", None),
            "last_score": None,
        }
        metric_hist = history.get(metric_name) if isinstance(history, dict) else None
        if metric_hist:
            try:
                objective_meta["last_score"] = float(metric_hist[-1])
            except Exception:
                pass
    _write_results(
        run_dir,
        history,
        best_epoch,
        include_history=include_history,
        objective_meta=objective_meta,
        test_metrics=test_metrics,
    )
    if _SKIP_ARTIFACTS:
        print("[run_trial] Artifact saving disabled for this run.")
    else:
        _save_best_state(run_dir, best_state)
        _save_plots(run_dir, history, best_epoch)
    _write_config(run_dir, cfg)


if __name__ == "__main__":
    main()
