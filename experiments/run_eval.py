#!/usr/bin/env python
"""
Run a fresh training/evaluation for the best config of an HPO study.

Heuristics:
- Picks the best trial by mean repeat score from experiments/hpo/<study>/repeats/repeats.json.
- Pulls the hyperparameters for that trial from experiments/hpo/<study>/topk.yaml.
- Reads the trial name from the saved HPO snapshot (snapshots/hpo.yaml).
- Retrains via experiments.run_trial with those overrides and writes outputs under experiments/eval/<study>/best_trial_<id>.
- Supports running multiple eval repeats (different seeds) via conf/eval/<study>.yaml or conf/eval/default.yaml.

Note: This retrains on the original train/val splits (no train+val merge implemented).
"""
from __future__ import annotations
import argparse
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _flatten_overrides(prefix: str, value: Any, exclude_roots: set[str]) -> list[str]:
    root = prefix.split(".", 1)[0] if prefix else prefix
    if root in exclude_roots:
        return []
    if isinstance(value, dict):
        items: list[str] = []
        for k, v in value.items():
            key_path = f"{prefix}.{k}" if prefix else k
            items.extend(_flatten_overrides(key_path, v, exclude_roots))
        return items
    return [f"{prefix}={json.dumps(value) if isinstance(value, (list, dict)) else value}"]


def _params_to_overrides(params: Dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    for k, v in params.items():
        overrides.extend(_flatten_overrides(k, v, set()))
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Evaluate best HPO config on train/val/test")
    parser.add_argument("--study-name", required=True, help="Study/run name under experiments/hpo/<study>")
    parser.add_argument("--env", required=True, help="Hydra env override (e.g., remote, local)")
    parser.add_argument("--hpo-root", type=str, default=None, help="Path to HPO study root (contains repeats/topk)")
    parser.add_argument("--trial", type=str, default=None, help="Optional explicit trial name to override inference")
    parser.add_argument("--eval-config", type=str, default=None, help="Optional path to eval config YAML")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=5, help="Number of eval repeats (seeds will increment)")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip already completed repeats")
    parser.add_argument("--epochs", type=int, default=None, help="Override trainer.epochs for eval runs")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--module", type=str, default="experiments.run_trial")
    args = parser.parse_args()

    if args.hpo_root:
        hpo_root = Path(args.hpo_root)
    else:
        hpo_root = Path("experiments") / "hpo" / args.study_name
    repeats_path = hpo_root / "repeats" / "repeats.json"
    topk_path = hpo_root / "topk.yaml"
    snapshot_hpo = hpo_root / "snapshots" / "hpo.yaml"
    # Backward compatibility: if snapshots missing, fall back to HPO search space file
    snapshot_trial = hpo_root / "snapshots" / "trial.yaml"
    fallback_hpo = hpo_root / "search_space.yaml"

    if not repeats_path.exists():
        raise FileNotFoundError(f"Repeats not found: {repeats_path}")
    if not topk_path.exists():
        raise FileNotFoundError(f"topk.yaml not found: {topk_path}")

    with repeats_path.open("r") as f:
        repeats = json.load(f).get("repeats", [])
    by_trial: Dict[int, list[float]] = {}
    for r in repeats:
        try:
            tnum = int(r.get("trial_number"))
            val = float(r.get("value"))
            by_trial.setdefault(tnum, []).append(val)
        except Exception:
            continue
    if not by_trial:
        raise RuntimeError("No repeat scores found.")
    best_trial = max(by_trial.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]

    with topk_path.open("r") as f:
        if yaml:
            topk = (yaml.safe_load(f) or {}).get("trials", [])
        else:
            topk = json.load(f).get("trials", [])
    params = None
    for entry in topk:
        if int(entry.get("trial_number", -1)) == best_trial:
            params = entry.get("params", {})
            break
    if params is None:
        raise RuntimeError(f"No params found in topk.yaml for trial {best_trial}")

    # Require a resolved config from repeats/trials for full overrides
    resolved_cfg = None
    resolved_paths = [
        hpo_root / "repeats" / f"trial_{best_trial:04d}_rep_0" / "resolved_config.yaml",
        hpo_root / "trials" / f"trial_{best_trial:04d}" / "resolved_config.yaml",
    ]
    for rp in resolved_paths:
        if rp.exists() and yaml is not None:
            try:
                with rp.open("r") as f:
                    resolved_cfg = yaml.safe_load(f) or {}
                    break
            except Exception:
                resolved_cfg = None
    if resolved_cfg is None:
        raise RuntimeError(f"No resolved_config.yaml found for trial {best_trial} in repeats/ or trials/")

    trial_name = None
    if snapshot_hpo.exists() and yaml is not None:
        try:
            with snapshot_hpo.open("r") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data.get("trial"), str):
                trial_name = data["trial"]
        except Exception:
            trial_name = None
    # Fallback to trial snapshot or search_space.yaml (legacy)
    if not trial_name and snapshot_trial.exists() and yaml is not None:
        try:
            with snapshot_trial.open("r") as f:
                tdata = yaml.safe_load(f) or {}
            if isinstance(tdata.get("trial_name"), str):
                trial_name = tdata["trial_name"]
        except Exception:
            trial_name = None
    if not trial_name and fallback_hpo.exists() and yaml is not None:
        try:
            with fallback_hpo.open("r") as f:
                fdata = yaml.safe_load(f) or {}
            if isinstance(fdata.get("trial"), str):
                trial_name = fdata["trial"]
        except Exception:
            trial_name = None
    if not trial_name and args.trial:
        trial_name = args.trial
    if not trial_name and resolved_cfg:
        try:
            tval = resolved_cfg.get("trial")
            if isinstance(tval, str):
                trial_name = tval
        except Exception:
            trial_name = None
    if not trial_name:
        raise RuntimeError("Could not determine trial name from snapshot or search_space.yaml")

    # Eval config (repeat counts)
    eval_cfg_path = None
    if args.eval_config:
        eval_cfg_path = Path(args.eval_config)
        if not eval_cfg_path.exists():
            raise FileNotFoundError(f"Eval config not found: {eval_cfg_path}")
    else:
        candidate = Path("conf") / "eval" / f"{args.study_name}.yaml"
        if candidate.exists():
            eval_cfg_path = candidate

    eval_cfg: Dict[str, Any] = {}
    if eval_cfg_path and yaml is not None:
        with eval_cfg_path.open("r") as f:
            eval_cfg = yaml.safe_load(f) or {}

    repeat_count = args.repeat_count
    if eval_cfg:
        repeat_count = int(
            eval_cfg.get("repeat", {}).get("count", eval_cfg.get("eval", {}).get("repeats", repeat_count))
        )

    base_eval_dir = Path(resolved_cfg.get("experiments_dir", "experiments")).expanduser()
    if not base_eval_dir.is_absolute():
        base_eval_dir = Path.cwd() / base_eval_dir
    eval_root = base_eval_dir / "eval" / args.study_name / f"best_trial_{best_trial:04d}"
    eval_root.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        args.python,
        "-m",
        args.module,
    ]
    base_overrides = [
        f"env={args.env}",
        f"trial={trial_name}",
        "run.group=eval",
        f"run.study={args.study_name}",
    ]

    # Merge train + val for final training; keep test as defined
    merged_cfg = deepcopy(resolved_cfg)
    try:
        data_cfg = merged_cfg.get("data", {})
        if isinstance(data_cfg, dict):
            train_subj = data_cfg.get("train_subjects", []) or []
            val_subj = data_cfg.get("val_subjects", []) or []
            test_subj = data_cfg.get("test_subjects", []) or data_cfg.get("test", [])
            merged_train = []
            for s in list(train_subj) + list(val_subj):
                if s not in merged_train:
                    merged_train.append(s)
            data_cfg["train_subjects"] = merged_train
            data_cfg["val_subjects"] = []
            if test_subj:
                data_cfg["test_subjects"] = test_subj
        trainer_cfg = merged_cfg.get("trainer", {})
        if isinstance(trainer_cfg, dict):
            trainer_cfg["disable_val"] = True
            if args.epochs is not None:
                trainer_cfg["epochs"] = int(args.epochs)
            merged_cfg["trainer"] = trainer_cfg
    except Exception:
        merged_cfg = resolved_cfg

    exclude = {"run", "cluster", "env", "seed"}
    extra_overrides = _flatten_overrides("", merged_cfg, exclude)
    runs: list[dict[str, Any]] = []

    for rep_idx in range(repeat_count):
        seed_val = args.seed + rep_idx
        run_dir = eval_root / f"rep_{rep_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        results_path = run_dir / "results.json"
        if not args.no_resume and results_path.exists():
            try:
                with results_path.open("r") as rf:
                    existing = json.load(rf)
                metrics = existing.get("final_metrics", {})
            except Exception:
                metrics = {}
            runs.append(
                {
                    "rep": rep_idx,
                    "seed": seed_val,
                    "run_dir": str(run_dir),
                    "final_metrics": metrics,
                    "skipped": True,
                }
            )
            continue
        cmd = base_cmd + base_overrides + extra_overrides + [
            f"seed={seed_val}",
            f"run.dir={str(run_dir)}",
        ]

        print("Running eval:", " ".join(shlex.quote(x) for x in cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"Eval run failed with code {proc.returncode}")

        metrics = {}
        results_path = run_dir / "results.json"
        if results_path.exists():
            try:
                with results_path.open("r") as rf:
                    metrics = json.load(rf).get("final_metrics", {})
            except Exception:
                metrics = {}
        runs.append(
            {
                "rep": rep_idx,
                "seed": seed_val,
                "run_dir": str(run_dir),
                "final_metrics": metrics,
            }
        )

    # Aggregate only the key test metrics we care about (F1, accuracy)
    target_metrics = ["test_f1", "test_acc"]
    aggregate: Dict[str, Dict[str, float]] = {}
    for key in target_metrics:
        vals: list[float] = []
        for r in runs:
            fm = r.get("final_metrics") or {}
            v = fm.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            continue
        avg = mean(vals)
        std = pstdev(vals) if len(vals) > 1 else 0.0
        aggregate[key] = {"mean": avg, "std": std, "count": len(vals)}

    summary = {
        "study": args.study_name,
        "trial_number": best_trial,
        "trial_name": trial_name,
        "repeat_count": repeat_count,
        "test_metrics": aggregate,
    }
    with (eval_root / "eval_summary.json").open("w") as sf:
        json.dump(summary, sf, indent=2)


if __name__ == "__main__":
    main()
