#!/usr/bin/env python
"""
Run a fresh training/evaluation for the best config of an HPO study.

Heuristics:
- Picks the best trial by mean repeat score from repeats_report.yaml (already sorted).
- Pulls the hyperparameters for that trial from repeats_best_params.yaml.
- Reads the trial name from the saved HPO snapshot (snapshots/hpo.yaml).
- Retrains via experiments.run_trial with base trial config + best params overrides.
- Outputs written to experiments/eval/<study>/best_trial_<id>/.
- Supports running multiple eval repeats (different seeds) for robust test estimates.

Config construction: conf/trial/<trial_name>.yaml + best_params overrides
"""
from __future__ import annotations
import argparse
import json
import shlex
import subprocess
import sys
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
    parser.add_argument("--output-root", type=str, default=None, help="Path to store eval results (default: experiments/eval/<study>)")
    parser.add_argument("--trial", type=str, default=None, help="Optional explicit trial name to override inference")
    parser.add_argument("--eval-config", type=str, default=None, help="Optional path to eval config YAML")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=10, help="Number of eval repeats (seeds will increment)")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip already completed repeats")
    parser.add_argument("--epochs", type=int, default=None, help="Override trainer.epochs for eval runs")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--module", type=str, default="experiments.run_trial")
    args = parser.parse_args()

    if args.hpo_root:
        hpo_root = Path(args.hpo_root)
    else:
        hpo_root = Path("experiments") / "hpo" / args.study_name
    
    best_params_path = hpo_root / "repeats_best_params.yaml"
    snapshot_hpo = hpo_root / "snapshots" / "hpo.yaml"
    # Backward compatibility: if snapshots missing, fall back to HPO search space file
    snapshot_trial = hpo_root / "snapshots" / "trial.yaml"
    fallback_hpo = hpo_root / "search_space.yaml"

    if not best_params_path.exists():
        raise FileNotFoundError(f"repeats_best_params.yaml not found: {best_params_path}")

    # Load best params (search space overrides only)
    with best_params_path.open("r") as f:
        if yaml:
            best_params = yaml.safe_load(f) or {}
        else:
            best_params = json.load(f)
    
    print(f"[eval] Loaded best params from: {best_params_path}")

    # Determine trial name from snapshots
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
    if not trial_name:
        raise RuntimeError("Could not determine trial name from snapshot or search_space.yaml. Use --trial to specify.")

    # Load the base trial config to get full training epochs (not HPO epochs)
    trial_cfg_path = Path("conf") / "trial" / f"{trial_name}.yaml"
    full_epochs = None
    if trial_cfg_path.exists() and yaml is not None:
        try:
            with trial_cfg_path.open("r") as f:
                trial_cfg_data = yaml.safe_load(f) or {}
            trainer_section = trial_cfg_data.get("trainer", {})
            if isinstance(trainer_section, dict):
                full_epochs = trainer_section.get("epochs")
        except Exception:
            pass

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

    # Determine eval output directory
    if args.output_root:
        eval_root = Path(args.output_root)
    else:
        eval_root = Path("experiments") / "eval" / args.study_name
    eval_root.mkdir(parents=True, exist_ok=True)
    print(f"[eval] Output directory: {eval_root}")

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

    # Add epochs override if specified
    epochs_overrides: list[str] = []
    if args.epochs is not None:
        epochs_overrides.append(f"trainer.epochs={args.epochs}")
    elif full_epochs is not None:
        epochs_overrides.append(f"trainer.epochs={full_epochs}")
        print(f"[eval] Using full training epochs from trial config: {full_epochs}")

    # Convert best_params to command-line overrides (search space params only)
    param_overrides = _params_to_overrides(best_params)
    print(f"[eval] Applying {len(param_overrides)} hyperparameter overrides from best trial")

    runs: list[dict[str, Any]] = []

    for rep_idx in range(repeat_count):
        seed_val = args.seed + rep_idx
        run_dir = eval_root / f"run_{rep_idx + 1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        results_path = run_dir / "results.json"
        if not args.no_resume and results_path.exists():
            try:
                with results_path.open("r") as rf:
                    existing = json.load(rf)
                metrics = existing.get("final_metrics", {})
            except Exception:
                metrics = {}
            print(f"[eval] Repeat {rep_idx + 1}/{repeat_count} already exists, skipping (use --no-resume to force)")
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
        cmd = base_cmd + base_overrides + param_overrides + epochs_overrides + [
            f"seed={seed_val}",
            f"run.dir={str(run_dir)}",
        ]

        print(f"\n[eval] === Repeat {rep_idx + 1}/{repeat_count} (seed={seed_val}) ===")
        print(f"[eval] Output dir: {run_dir}")
        print("[eval] Running:", " ".join(shlex.quote(x) for x in cmd))
        sys.stdout.flush()

        # Stream output in real-time instead of buffering
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()

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

    # Aggregate test metrics (expand to include more metrics if available)
    target_metrics = ["test_f1", "test_acc", "test_precision", "test_recall", "test_loss"]
    # Also collect any other test_* metrics dynamically
    all_test_keys: set[str] = set(target_metrics)
    for r in runs:
        fm = r.get("final_metrics") or {}
        for k in fm.keys():
            if k.startswith("test_"):
                all_test_keys.add(k)
    
    aggregate: Dict[str, Dict[str, float]] = {}
    for key in sorted(all_test_keys):
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
        "trial_name": trial_name,
        "best_params": best_params,
        "repeat_count": repeat_count,
        "test_metrics": aggregate,
    }
    summary_path = eval_root / "eval_summary.json"
    with summary_path.open("w") as sf:
        json.dump(summary, sf, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print(f"[eval] EVALUATION COMPLETE")
    print(f"[eval] Study: {args.study_name}")
    print(f"[eval] Repeats: {repeat_count}")
    print(f"[eval] Results saved to: {eval_root}/")
    print(f"[eval] Summary: {summary_path}")
    print("-" * 60)
    for key, stats in sorted(aggregate.items()):
        print(f"[eval] {key}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
