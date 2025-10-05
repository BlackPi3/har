#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Optimization orchestrator.

Each Optuna trial launches the existing single-run entrypoint
`experiments/run_experiment.py` with Hydra overrides and reads its
results.json to obtain the objective metric.

Example usage:
  # Fast search on debug split (maximize best_val_f1)
  python experiments/run_hpo.py \
    --n-trials 20 --metric val_f1 --direction maximize \
    --study-name mmfit_debug_sc2 \
    --storage experiments/outputs/optuna/mmfit_debug_sc2.db \
    data=mmfit_debug trainer.epochs=5

  # With W&B (optional; pass your own overrides)
  python experiments/run_hpo.py --n-trials 10 --metric val_f1 --direction maximize \
    trainer.logger=wandb trainer.wandb.enabled=true trainer.wandb.group=optuna-sc2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import subprocess

import optuna


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = REPO_ROOT / "experiments" / "run_experiment.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Optuna HPO by invoking run_experiment.py per trial")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides passed through to run_experiment.py (e.g., data=mmfit_debug trainer.epochs=5)")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--study-name", type=str, default="har_hpo")
    p.add_argument("--storage", type=str, default=None, help="Path to SQLite DB (e.g., experiments/outputs/optuna/study.db). If omitted, uses in-memory study.")
    p.add_argument("--metric", choices=["val_f1", "val_loss"], default="val_f1")
    p.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", type=str, default="experiments/outputs/hpo", help="Root folder for trial outputs (relative to repo)")
    p.add_argument("--search-space", type=str, default=str(REPO_ROOT / "conf/hpo/scenario2_mmfit.yaml"), help="Path to YAML file defining Optuna search space")
    return p.parse_args()


def build_trial_dir(output_root: Path, study_name: str, trial_number: int) -> Path:
    return output_root / study_name / f"trial_{trial_number:03d}"


def load_search_space(path: Path) -> Dict[str, Any]:
    import yaml
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "search_space" not in data:
        raise ValueError("Search space YAML must contain a top-level 'search_space' mapping")
    return data["search_space"]


def suggest_params_generic(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample parameters based on a generic YAML-defined search space.

    Spec schema per key:
      type: float|int|categorical|bool
      low: <number> (for float/int)
      high: <number> (for float/int)
      log: true|false (optional for float/int)
      step: <number> (optional, for int and float)
      choices: [..] (for categorical)
      tie_to: <other_key> (copy value from other_key; no sampling)
    """
    sampled: Dict[str, Any] = {}
    # First pass: sample untied params
    for key, spec in space.items():
        if spec is None:
            continue
        if isinstance(spec, dict) and spec.get("tie_to"):
            # Will handle in second pass
            continue
        t = spec.get("type") if isinstance(spec, dict) else None
        if t == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            log = bool(spec.get("log", False))
            step = spec.get("step", None)
            if step is not None:
                sampled[key] = trial.suggest_float(key, low, high, log=log, step=float(step))
            else:
                sampled[key] = trial.suggest_float(key, low, high, log=log)
        elif t == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            sampled[key] = trial.suggest_int(key, low, high, step=step)
        elif t == "categorical":
            choices = spec["choices"]
            sampled[key] = trial.suggest_categorical(key, choices)
        elif t == "bool":
            sampled[key] = trial.suggest_categorical(key, [True, False])
        else:
            raise ValueError(f"Unsupported type for '{key}': {t}")

    # Second pass: apply ties
    for key, spec in space.items():
        if isinstance(spec, dict) and spec.get("tie_to"):
            ref = spec["tie_to"]
            if ref not in sampled:
                raise ValueError(f"tie_to references unknown key: {ref}")
            sampled[key] = sampled[ref]

    return sampled


def run_single(cfg_overrides: List[str], trial_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Invoke run_experiment.py with overrides; return (success, metrics_dict)."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    # Force Hydra to use a fixed trial dir to simplify retrieval
    hydra_override = f"hydra.run.dir={trial_dir.as_posix()}"
    cmd = [sys.executable, str(RUN_EXPERIMENT)] + cfg_overrides + [hydra_override]
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = proc.stdout
        # Persist stdout to trial dir for debugging
        (trial_dir / "stdout.log").write_text(output)
        if proc.returncode != 0:
            return False, {"error": f"returncode={proc.returncode}"}
        results_path = trial_dir / "results.json"
        if not results_path.exists():
            return False, {"error": "results.json not found"}
        data = json.loads(results_path.read_text())
        metrics = data.get("final_metrics", {})
        return True, metrics
    except Exception as e:
        return False, {"error": str(e)}


def main() -> None:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    study_storage = None
    if args.storage:
        storage_path = (REPO_ROOT / args.storage).resolve()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        study_storage = f"sqlite:///{storage_path.as_posix()}"

    study = optuna.create_study(direction=args.direction, study_name=args.study_name, storage=study_storage, load_if_exists=True)

    # Load search space (YAML). If not found, fall back to internal defaults
    search_space: Dict[str, Any]
    space_path = Path(args.search_space)
    if space_path.exists():
        search_space = load_search_space(space_path)
        print(f"✓ Loaded search space from {space_path}")
    else:
        print(f"! Search space file not found at {space_path}; using minimal defaults")
        search_space = {
            "optim.lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "optim.weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
            "experiment.alpha": {"type": "float", "low": 0.5, "high": 2.0},
            "experiment.beta": {"type": "float", "low": 0.0, "high": 1.0},
        }

    def objective(trial: optuna.Trial) -> float:
        # Compose overrides: user overrides + suggested params + seed for reproducibility
        trial_params = suggest_params_generic(trial, search_space)
        param_overrides = [f"{k}={v}" for k, v in trial_params.items()]
        # Propagate seed to ensure fairness across trials
        seed_override = f"seed={args.seed}"
        # Trial-specific run dir
        trial_dir = build_trial_dir(output_root, args.study_name, trial.number)
        # Collect full override list
        overrides = list(args.overrides) + param_overrides + [seed_override]

        success, metrics = run_single(overrides, trial_dir)
        if not success:
            # Mark as failed with a large penalty depending on direction
            if args.direction == "maximize":
                return -1e9
            else:
                return 1e9

        # Choose objective
        if args.metric == "val_f1":
            val = metrics.get("best_val_f1") or metrics.get("val_f1_last")
            if val is None:
                # Unable to read metric, treat as failed
                return -1e9 if args.direction == "maximize" else 1e9
            return float(val)
        else:  # val_loss
            val = metrics.get("best_val_loss") or metrics.get("val_loss_last")
            if val is None:
                return 1e9 if args.direction == "minimize" else -1e9
            # If maximizing val_loss (unlikely), invert logic
            return float(val)

    study.optimize(objective, n_trials=args.n_trials)

    # Persist study summary
    summary_dir = output_root / args.study_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    best = study.best_trial
    summary = {
        "study_name": args.study_name,
        "direction": args.direction,
        "metric": args.metric,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
    }
    (summary_dir / "best.json").write_text(json.dumps(summary, indent=2))

    # Export trials as CSV
    csv_path = summary_dir / "trials.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "state", "value", "params_json"])
        for t in study.trials:
            writer.writerow([t.number, str(t.state), t.value, json.dumps(t.params)])

    print(f"Optuna finished. Best {args.metric} = {best.value:.6f}")
    print(f"Best params: {best.params}")
    print(f"Study saved under: {summary_dir}")


if __name__ == "__main__":
    main()
