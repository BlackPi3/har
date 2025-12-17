#!/usr/bin/env python
"""
Run repeat evaluations for the top-k configs found by run_hpo.

Usage:
    python experiments/run_topk.py \
        --study-name scenario2_mmfit \
        --space-config conf/hpo/scenario2_mmfit.yaml \
        --env remote \
        --base-seed 0

This script:
    - Reads top_k.json from experiments/hpo/<study-name>/ (or --topk-source-root)
    - Reads repeat.k from the HPO YAML (conf/hpo/<study>.yaml) to decide how many repeats
    - Runs experiments.run_trial for each top-k config, with distinct seeds per repeat
    - Stores results under experiments/top_k/<study-name>/repeats/trial_<id>/rep_<i>/
"""
from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config import merge_dicts

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return str(value)


def _flatten_overrides(prefix: str, value: Any) -> list[str]:
    if isinstance(value, dict):
        items: list[str] = []
        for key, val in value.items():
            key_path = f"{prefix}.{key}" if prefix else key
            items.extend(_flatten_overrides(key_path, val))
        return items
    return [f"{prefix}={_format_override_value(value)}"]


def _flatten_cfg_for_overrides(cfg: Dict[str, Any], exclude: Tuple[str, ...] = ("run",)) -> list[str]:
    items: list[str] = []
    for key, val in cfg.items():
        if key in exclude:
            continue
        items.extend(_flatten_overrides(key, val))
    return items


def _parse_hpo_meta(yaml_path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "study_name": None,
        "trial": None,
        "trainer_overrides": [],
        "data_overrides": [],
        "repeat_k": 1,
    }
    if yaml is None:
        return meta
    try:
        with yaml_path.open("r") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data.get("study_name"), str):
            meta["study_name"] = data.get("study_name")
        sweeper = (data.get("hydra", {}) or {}).get("sweeper", {}) or {}
        if isinstance(sweeper, dict):
            if isinstance(sweeper.get("study_name"), str):
                if "${" not in sweeper.get("study_name"):
                    meta["study_name"] = sweeper.get("study_name")
        if isinstance(data.get("trial"), str):
            meta["trial"] = data.get("trial")
        trainer_node = data.get("trainer")
        if trainer_node is not None:
            if isinstance(trainer_node, str):
                trainer_node = {"name": trainer_node}
            if isinstance(trainer_node, dict):
                meta["trainer_overrides"] = _flatten_overrides("trainer", trainer_node)
        data_node = data.get("data")
        if data_node is not None:
            if isinstance(data_node, str):
                data_node = {"name": data_node}
            if isinstance(data_node, dict):
                meta["data_overrides"] = _flatten_overrides("data", data_node)
        repeat_node = data.get("repeat", {}) or {}
        if isinstance(repeat_node, dict):
            try:
                k_val = repeat_node.get("count", repeat_node.get("k", None))
                if isinstance(k_val, int) and k_val > 0:
                    meta["repeat_k"] = k_val
            except Exception:
                pass
    except Exception:
        pass
    return meta


def _load_trial_default_epochs(trial_name: str | None) -> int | None:
    if not trial_name:
        return None
    trial_path = Path("conf/trial") / f"{trial_name}.yaml"
    if not trial_path.exists():
        return None
    try:
        with trial_path.open("r") as f:
            data = yaml.safe_load(f) or {}
        trainer = data.get("trainer") if isinstance(data, dict) else None
        if isinstance(trainer, dict):
            epochs = trainer.get("epochs")
            if isinstance(epochs, int) and epochs > 0:
                return epochs
    except Exception:
        return None
    return None


def _load_trial_cfg(trial_name: str | None) -> Dict[str, Any]:
    if not trial_name or yaml is None:
        return {}
    trial_path = Path("conf/trial") / f"{trial_name}.yaml"
    if not trial_path.exists():
        return {}
    try:
        with trial_path.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _params_to_overrides(params: Dict[str, Any]) -> list[str]:
    return [f"{k}={_format_override_value(v)}" for k, v in params.items()]


def _params_to_nested_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for key, val in (params or {}).items():
        parts = key.split(".")
        d = nested
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = val
    return nested


def _score_from_results(results: Dict[str, Any], metric: str):
    objective_info = (results or {}).get("objective", {}) or {}
    score = objective_info.get("best_score")
    if score is not None:
        try:
            return float(score)
        except Exception:
            pass
    fm = (results or {}).get("final_metrics", {}) or {}
    # Default to val_f1 if metric missing
    val = None
    if metric == "val_f1":
        val = fm.get("val_f1_last") or fm.get("best_val_f1")
    else:
        val = fm.get("val_loss_last") or fm.get("best_val_loss")
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def run_trial(cmd_base: list[str], run_dir: Path, group: str = "topk") -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"run.dir={str(run_dir)}",
        f"run.group={group}",
    ]
    full_cmd = cmd_base + overrides
    print("Launching:", " ".join(shlex.quote(p) for p in full_cmd))
    proc = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {}
    try:
        with results_path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run repeats for top-k HPO configs")
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--space-config", type=str, required=True, help="conf/hpo/<study>.yaml (for repeat.k and trial defaults)")
    parser.add_argument("--topk-source-root", type=str, default=None,
                        help="Root where top_k.yaml lives (default: experiments/top_k/<study-name>, falls back to experiments/hpo/<study-name>)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Destination root for repeat runs (default: experiments/top_k/<study-name>)")
    parser.add_argument("--env", type=str, required=True, help="Hydra env override (e.g., local, remote)")
    parser.add_argument("--base-seed", type=int, default=0, help="Base seed; each repeat adds an offset for uniqueness")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--module", type=str, default="experiments.run_trial")
    parser.add_argument("--dry", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    space_yaml_path = Path(args.space_config)
    if not space_yaml_path.exists():
        raise FileNotFoundError(f"--space-config not found: {space_yaml_path}")
    meta = _parse_hpo_meta(space_yaml_path)

    study_name = meta.get("study_name") or args.study_name
    if not meta.get("trial"):
        raise RuntimeError("No trial name found in the HPO YAML. Set 'trial:' in conf/hpo/<study>.yaml.")

    if args.topk_source_root:
        topk_root_candidates = [Path(args.topk_source_root)]
    else:
        topk_root_candidates = [
            Path("experiments") / "top_k" / study_name,
            Path("experiments") / "hpo" / study_name,
        ]
    topk_root = None
    topk_yaml = None
    for cand in topk_root_candidates:
        cand_y = cand / "top_k.yaml"
        if cand_y.exists():
            topk_root = cand
            topk_yaml = cand_y
            break
    if topk_yaml is None or topk_root is None:
        raise FileNotFoundError("top_k.yaml not found. Run HPO first.")
    with topk_yaml.open("r") as f:
        topk_payload = yaml.safe_load(f) if yaml else json.load(f)
    trials: List[Dict[str, Any]] = topk_payload.get("trials", [])
    metric = topk_payload.get("metric", "val_f1")

    repeat_k = max(1, int(meta.get("repeat_k") or 1))
    default_trial_epochs = _load_trial_default_epochs(meta.get("trial"))
    trial_base_cfg = _load_trial_cfg(meta.get("trial"))

    out_root = Path(args.output_root) if args.output_root else Path("experiments") / "top_k" / study_name
    repeats_root = out_root / "repeats"
    trial_export_root = out_root / "trials"
    out_root.mkdir(parents=True, exist_ok=True)

    base_cmd = [args.python, "-m", args.module]

    repeat_summary = []
    for entry in trials:
        tnum = entry.get("trial_number")
        params = entry.get("params", {})
        # Prefer resolved config snapshot if present to ensure exact settings.
        resolved_cfg_path = topk_root / "trials" / f"trial_{tnum:04d}" / "resolved_config.yaml"
        resolved_cfg = None
        if resolved_cfg_path.exists() and yaml is not None:
            try:
                with resolved_cfg_path.open("r") as f:
                    resolved_cfg = yaml.safe_load(f) or {}
            except Exception:
                resolved_cfg = None
        params_overrides: List[str]
        if resolved_cfg:
            resolved_cfg["seed"] = None  # will override below
            # Restore trial default epochs for repeat runs if available
            if default_trial_epochs:
                resolved_cfg.setdefault("trainer", {})
                if isinstance(resolved_cfg["trainer"], dict):
                    resolved_cfg["trainer"]["epochs"] = default_trial_epochs
            params_overrides = _flatten_cfg_for_overrides(resolved_cfg)
        else:
            params_overrides = _params_to_overrides(params)
        # Ensure we don't carry forward the original seed; each repeat gets its own seed below
        params_overrides = [ov for ov in params_overrides if not ov.startswith("seed=")]

        # Export merged trial config + params for clarity
        merged_cfg = merge_dicts(trial_base_cfg or {}, _params_to_nested_dict(params))
        export_dir = trial_export_root / f"trial_{tnum:04d}"
        export_dir.mkdir(parents=True, exist_ok=True)
        if merged_cfg and yaml is not None:
            try:
                with (export_dir / "merged_config.yaml").open("w") as f:
                    yaml.safe_dump(merged_cfg, f, sort_keys=False)
            except Exception:
                pass
        for rep_idx in range(repeat_k):
            seed = args.base_seed + tnum * repeat_k + rep_idx
            run_overrides = params_overrides + [f"seed={seed}", f"env={args.env}", f"trial={meta['trial']}"]
            rep_dir = repeats_root / f"trial_{tnum:04d}" / f"rep_{rep_idx}"
            cmd = base_cmd + run_overrides
            if args.dry:
                print("DRY RUN:", " ".join(shlex.quote(x) for x in cmd), "->", rep_dir)
                continue
            results = run_trial(cmd, rep_dir, group="topk")
            score = _score_from_results(results, metric)
            repeat_summary.append(
                {
                    "trial_number": tnum,
                    "repeat_index": rep_idx,
                    "seed": seed,
                    "value": score,
                    "run_dir": str(rep_dir),
                }
            )

    if repeat_summary and not args.dry:
        try:
            with (out_root / "repeats.json").open("w") as f:
                json.dump(
                    {
                        "metric": metric,
                        "repeat_k": repeat_k,
                        "repeats": repeat_summary,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"[topk] Failed to write repeats.json: {e}")


if __name__ == "__main__":
    main()
