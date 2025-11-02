#!/usr/bin/env python
"""
Native Optuna orchestrator (no Hydra Sweeper).

Runs experiments/run_trial.py in a subprocess per trial with Hydra overrides,
reads results.json from the per-trial run dir, and reports the chosen metric
to Optuna. Designed to work on both local and SLURM/containers.

Usage examples:
  python experiments/run_optuna.py \
    --n-trials 10 \
    --study-name scenario2_mmfit \
    --storage /netscratch/$USER/experiments/output/scenario2_mmfit/scenario2_mmfit.db \
    --metric val_f1 --direction maximize \
    --output-root /netscratch/$USER/experiments/output/scenario2_mmfit \
    -- \
    env=remote data=mmfit scenario=scenario2 trainer.epochs=5

Notes:
    - Arguments after "--" are forwarded as Hydra overrides to experiments.run_trial.
  - If --storage is a filesystem path, it's converted to a SQLite URL.
  - Each trial runs under: <output_root>/trial_<N>/
"""
from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

try:
    import optuna
    from optuna.pruners import MedianPruner
except Exception:  # pragma: no cover
    print("Optuna is required. Install with: pip install optuna", file=sys.stderr)
    raise


def to_sqlite_url(storage: str) -> str:
    if storage.startswith("sqlite://"):
        return storage
    p = Path(storage)
    if not p.is_absolute():
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:////{p}"


def default_output_root(study: str) -> Path:
    ns = os.environ.get("USER") or "user"
    base = Path(f"/netscratch/{ns}/experiments/output/{study}")
    return base


def build_space(trial: optuna.trial.Trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    # Optimizer
    params["optim.lr"] = trial.suggest_float("optim.lr", 1e-5, 1e-2, log=True)
    params["optim.weight_decay"] = trial.suggest_float("optim.weight_decay", 1e-6, 1e-2, log=True)
    # Scenario loss weights
    params["scenario.alpha"] = trial.suggest_float("scenario.alpha", 0.0, 10.0)
    params["scenario.beta"] = trial.suggest_float("scenario.beta", 0.0, 10.0)
    # Data / model coupling
    params["data.sensor_window_length"] = trial.suggest_categorical(
        "data.sensor_window_length", [128, 192, 256, 320, 384, 448, 512]
    )
    params["data.stride_seconds"] = trial.suggest_categorical(
        "data.stride_seconds", [0.1, 0.2, 0.5, 1.0]
    )
    params["model.feature_extractor.n_filters"] = trial.suggest_int(
        "model.feature_extractor.n_filters", 8, 32, step=4
    )
    params["model.feature_extractor.filter_size"] = trial.suggest_categorical(
        "model.feature_extractor.filter_size", [3, 5, 7]
    )
    # Throughput
    params["data.batch_size"] = trial.suggest_categorical(
        "data.batch_size", [1024, 1536, 2048]
    )
    return params


def run_trial(cmd_base: list[str], run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    # Force Hydra run dir and ensure chdir behavior
    overrides = [f"hydra.run.dir={str(run_dir)}"]
    full_cmd = cmd_base + overrides
    print("Launching:", " ".join(shlex.quote(p) for p in full_cmd))
    proc = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {}
    with results_path.open("r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Native Optuna HPO Orchestrator")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--study-name", type=str, default="scenario2_mmfit")
    parser.add_argument("--storage", type=str, default=None,
                        help="SQLite URL or filesystem path to .db (converted to sqlite URL if path)")
    parser.add_argument("--metric", type=str, default="val_f1", choices=["val_f1", "val_loss"])
    parser.add_argument("--direction", type=str, default=None, choices=["maximize", "minimize"])  # auto if None
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--module", type=str, default="experiments.run_trial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prune", action="store_true", help="Enable MedianPruner")
    parser.add_argument("--dry", action="store_true", help="Print commands without running")
    parser.add_argument("--", dest="sep", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides after --")

    args = parser.parse_args()

    storage_url = to_sqlite_url(args.storage) if args.storage else None
    if storage_url is None:
        out_base = args.output_root or str(default_output_root(args.study_name))
        db_path = Path(out_base) / f"{args.study_name}.db"
        storage_url = to_sqlite_url(str(db_path))

    out_root = Path(args.output_root) if args.output_root else default_output_root(args.study_name)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.direction is None:
        direction = "maximize" if args.metric == "val_f1" else "minimize"
    else:
        direction = args.direction

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = MedianPruner() if args.prune else None
    study = optuna.create_study(direction=direction, study_name=args.study_name,
                                storage=storage_url, load_if_exists=True, sampler=sampler, pruner=pruner)

    base_cmd = [args.python, "-m", args.module]
    # Default overrides if none provided
    hydra_overrides = args.overrides if args.overrides else [
        "env=remote", "data=mmfit", "scenario=scenario2", f"seed={args.seed}",
    ]

    def objective(trial: optuna.trial.Trial) -> float:
        params = build_space(trial)
        trial_dir = out_root / f"trial_{trial.number:04d}"

        # Convert params dict to hydra-style overrides
        trial_overrides = hydra_overrides + [f"{k}={v}" for k, v in params.items()]
        cmd = base_cmd + trial_overrides

        if args.dry:
            print("DRY RUN:", " ".join(shlex.quote(x) for x in cmd))
            return 0.0

        results = run_trial(cmd, trial_dir)
        if not results:
            # Failed run; assign a bad score so it's pruned from consideration
            return -1e9 if direction == "maximize" else 1e9

        fm = (results or {}).get("final_metrics", {})
        if args.metric == "val_f1":
            val = fm.get("val_f1_last") or fm.get("best_val_f1")
            if val is None:
                return -1e9
            return float(val)
        else:
            val = fm.get("val_loss_last") or fm.get("best_val_loss")
            if val is None:
                return 1e9
            # Optuna expects minimize for val_loss; if user selected maximize, invert
            return float(val)

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # Save quick summary
    best = study.best_trial
    summary = {
        "best_value": best.value,
        "best_params": best.params,
        "direction": direction,
        "metric": args.metric,
        "storage": storage_url,
        "study_name": args.study_name,
    }
    with (out_root / "best.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Trials CSV
    try:
        import csv
        with (out_root / "trials.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["number", "value", "state", "params"])
            for t in study.trials:
                writer.writerow([t.number, t.value, str(t.state), json.dumps(t.params)])
    except Exception:
        pass

    print("HPO complete. Best:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
