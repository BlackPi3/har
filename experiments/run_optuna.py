#!/usr/bin/env python
"""
Native Optuna orchestrator (no Hydra Sweeper).

Runs experiments/run_trial.py in a subprocess per trial with Hydra overrides,
reads results.json from the per-trial run dir, and reports the chosen metric
to Optuna. Designed to work on both local and SLURM/containers.

Search space source of truth: conf/hpo/*.yaml
    - Provide explicitly via --space-config conf/hpo/<name>.yaml, OR
    - Set HPO=<name> (the tool will auto-pick conf/hpo/$HPO.yaml), OR
    - As a convenience, if conf/hpo/<study-name>.yaml exists, it will be used.

Usage examples:
    python experiments/run_optuna.py \
        --n-trials 10 \
        --study-name scenario2_mmfit \
        --space-config conf/hpo/scenario2_mmfit.yaml \
        --metric val_f1 --direction maximize \
        --output-root experiments/hpo/scenario2_mmfit \
        --env remote --data mmfit --epochs 5

Notes:
    - Arguments after "--" are forwarded as Hydra overrides to experiments.run_trial.
    - If --storage is a filesystem path, it's converted to a SQLite URL.
    - Each trial runs under: <output_root>/trials/trial_<N>/
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

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover
    yaml = None


def to_sqlite_url(storage: str) -> str:
    if storage.startswith("sqlite://"):
        return storage
    p = Path(storage)
    if not p.is_absolute():
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:////{p}"


def default_output_root(study: str) -> Path:
    return Path("experiments") / "hpo" / study


def build_space(trial: optuna.trial.Trial) -> Dict[str, Any]:
    # Deprecated: search space must come from conf/hpo YAML.
    raise RuntimeError("Search space must be provided via --space-config (conf/hpo/*.yaml) or HPO env var.")


def _parse_number(token: str):
    t = token.strip()
    try:
        # Try int first where appropriate
        if "." not in t and "e" not in t and "E" not in t:
            return int(t)
        return float(t)
    except Exception:
        # Fallback to raw string
        return t


def _parse_sweeper_spec(spec: str, trial: optuna.trial.Trial):
    s = spec.strip().replace(" ", "")
    # tag(log, interval(a,b))
    if s.startswith("tag(log,interval("):
        # Expecting trailing "))" (closing interval and tag). Be resilient if only one ")" present.
        if s.endswith("))"):
            inner = s[len("tag(log,interval("):-2]
        elif s.endswith(")"):
            inner = s[len("tag(log,interval("):-1]
        else:
            inner = s[len("tag(log,interval(") :]
        lo, hi = inner.split(",")
        return lambda name: trial.suggest_float(name, float(lo), float(hi), log=True)
    # interval(a,b)
    if s.startswith("interval(") and s.endswith(")"):
        inner = s[len("interval("):-1]
        lo, hi = inner.split(",")
        return lambda name: trial.suggest_float(name, float(lo), float(hi))
    # range(start,stop,step)
    if s.startswith("range(") and s.endswith(")"):
        inner = s[len("range("):-1]
        parts = [p for p in inner.split(",") if p]
        if len(parts) == 3:
            start, stop, step = map(int, parts)
            # Hydra's range is like Python range (stop exclusive). Build explicit choices to avoid inclusivity confusion.
            values = list(range(start, stop, step))
            return lambda name: trial.suggest_categorical(name, values)
    # choice(v1,v2,...)
    if s.startswith("choice(") and s.endswith(")"):
        inner = s[len("choice("):-1]
        raw_vals = [v for v in inner.split(",") if v]
        values = [_parse_number(v) for v in raw_vals]
        return lambda name: trial.suggest_categorical(name, values)
    # Fallback: treat as categorical single value
    return lambda name: trial.suggest_categorical(name, [spec])


def build_space_from_yaml(trial: optuna.trial.Trial, yaml_path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not available; cannot load --space-config")
    with yaml_path.open("r") as f:
        data = yaml.safe_load(f)
    # Expect Hydra sweeper style at hydra.sweeper.params
    params_node = None
    try:
        params_node = data.get("hydra", {}).get("sweeper", {}).get("params", {})
    except Exception:
        params_node = {}
    if not isinstance(params_node, dict) or not params_node:
        raise ValueError(f"No params found in YAML sweeper file: {yaml_path}")
    suggested: Dict[str, Any] = {}
    for key, spec in params_node.items():
        suggest_fn = _parse_sweeper_spec(str(spec), trial)
        val = suggest_fn(key)
        suggested[key] = val
    return suggested


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


def _extract_yaml_meta(yaml_path: Path) -> Dict[str, Any]:
    """Extract optional metadata from conf/hpo YAML.
    Returns keys: study_name (str|None), metric (str|None), mode (str|None)
    """
    meta: Dict[str, Any] = {"study_name": None, "metric": None, "mode": None}
    if yaml is None:
        return meta
    try:
        with yaml_path.open("r") as f:
            data = yaml.safe_load(f) or {}
        # hydra.sweeper.study_name
        sweeper = (data.get("hydra", {}) or {}).get("sweeper", {}) or {}
        if isinstance(sweeper, dict):
            if isinstance(sweeper.get("study_name"), str):
                meta["study_name"] = sweeper.get("study_name")
        # hpo.metric/mode
        hpo_node = data.get("hpo", {}) or {}
        if isinstance(hpo_node, dict):
            if isinstance(hpo_node.get("metric"), str):
                meta["metric"] = hpo_node.get("metric")
            if isinstance(hpo_node.get("mode"), str):
                meta["mode"] = hpo_node.get("mode")
    except Exception:
        pass
    return meta


def _resolve_space_yaml(args) -> Path:
    # 1) explicit flag
    if args.space_config:
        p = Path(args.space_config)
        if not p.exists():
            raise FileNotFoundError(f"--space-config not found: {p}")
        return p
    # 2) env var HPO -> conf/hpo/$HPO.yaml
    env_hpo = os.environ.get("HPO")
    if env_hpo:
        p = Path("conf/hpo") / f"{env_hpo}.yaml"
        if p.exists():
            return p
    # 3) study-name heuristic
    if args.study_name:
        p = Path("conf/hpo") / f"{args.study_name}.yaml"
        if p.exists():
            return p
    raise FileNotFoundError(
        "No search space YAML found. Provide --space-config conf/hpo/<name>.yaml or set HPO=<name> (with conf/hpo/$HPO.yaml present)."
    )


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
    parser.add_argument("--space-config", type=str, default=None, help="Hydra sweeper YAML defining the search space (conf/hpo/*.yaml)")
    parser.add_argument("--dry", action="store_true", help="Print commands without running")
    # Explicit run-time configuration (avoid free-form overrides for simplicity)
    parser.add_argument("--env", type=str, default="remote", help="Hydra env choice (e.g., local, remote)")
    parser.add_argument("--data", type=str, default="mmfit", help="Dataset config choice (e.g., mmfit, mmfit_debug)")
    parser.add_argument("--epochs", type=int, default=None, help="Trainer epochs override; if omitted, use config default")

    args = parser.parse_args()

    # Resolve space YAML first and extract metadata to drive study/metric
    space_yaml_path = _resolve_space_yaml(args)
    meta = _extract_yaml_meta(space_yaml_path)

    # Adopt study name from YAML if provided
    study_name = meta.get("study_name") or args.study_name

    # Determine metric and direction; prefer YAML if present
    metric = meta.get("metric") or args.metric
    if args.direction is None:
        direction = (meta.get("mode") or ("maximize" if metric == "val_f1" else "minimize"))
    else:
        direction = args.direction

    # Storage and output paths (based on resolved study_name unless overridden)
    storage_url = to_sqlite_url(args.storage) if args.storage else None
    if storage_url is None:
        out_base = args.output_root or str(default_output_root(study_name))
        db_path = Path(out_base) / f"{study_name}.db"
        storage_url = to_sqlite_url(str(db_path))

    out_root = Path(args.output_root) if args.output_root else default_output_root(study_name)
    out_root.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = MedianPruner() if args.prune else None
    study = optuna.create_study(direction=direction, study_name=study_name,
                                storage=storage_url, load_if_exists=True, sampler=sampler, pruner=pruner)

    base_cmd = [args.python, "-m", args.module]
    # Build explicit Hydra overrides from flags
    hydra_overrides = [
        f"env={args.env}",
        f"data={args.data}",
        f"seed={args.seed}",
    ]
    if args.epochs is not None:
        hydra_overrides.append(f"trainer.epochs={args.epochs}")

    # YAML path already resolved above; fail-fast happened earlier

    def objective(trial: optuna.trial.Trial) -> float:
        params = build_space_from_yaml(trial, space_yaml_path)
        trial_dir = out_root / "trials" / f"trial_{trial.number:04d}"

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
        if metric == "val_f1":
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
        "metric": metric,
        "storage": storage_url,
        "study_name": study_name,
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
