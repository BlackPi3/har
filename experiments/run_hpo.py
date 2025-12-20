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
    python experiments/run_hpo.py \
        --n-trials 10 \
        --study-name scenario2_mmfit \
        --space-config conf/hpo/scenario2_mmfit.yaml \
        --metric val_f1 --direction maximize \
        --output-root experiments/hpo/scenario2_mmfit \
        --env remote --epochs 5

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
import shutil
import subprocess
import sys
from collections import Counter
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


def _mode_to_direction(mode: str | None) -> str | None:
    if not mode:
        return None
    m = mode.lower()
    if m in ("max", "maximize"):
        return "maximize"
    if m in ("min", "minimize"):
        return "minimize"
    return None


def build_space(trial: optuna.trial.Trial) -> Dict[str, Any]:
    # Deprecated: search space must come from conf/hpo YAML.
    raise RuntimeError("Search space must be provided via --space-config (conf/hpo/*.yaml) or HPO env var.")


def _parse_number(token: str):
    t = token.strip()
    if yaml is not None:
        try:
            return yaml.safe_load(t)
        except Exception:
            pass
    try:
        # Try int first where appropriate
        if "." not in t and "e" not in t and "E" not in t:
            return int(t)
        return float(t)
    except Exception:
        pass
    mappings = {"true": True, "false": False, "null": None}
    return mappings.get(t.lower(), t)


def _split_args(inner: str):
    args = []
    depth = 0
    current = []
    brackets = {"(": ")", "[": "]", "{": "}"}
    closing = {")", "]", "}"}
    for ch in inner:
        if ch in brackets:
            depth += 1
        elif ch in closing:
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


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
        lo_f, hi_f = float(lo), float(hi)
        if lo_f > hi_f:
            lo_f, hi_f = hi_f, lo_f
        return lambda name: trial.suggest_float(name, lo_f, hi_f, log=True)
    # interval(a,b)
    if s.startswith("interval(") and s.endswith(")"):
        inner = s[len("interval("):-1]
        lo, hi = inner.split(",")
        lo_f, hi_f = float(lo), float(hi)
        if lo_f > hi_f:
            lo_f, hi_f = hi_f, lo_f
        return lambda name: trial.suggest_float(name, lo_f, hi_f)
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
        raw_vals = _split_args(inner)
        parsed_vals = [_parse_number(v) for v in raw_vals]
        needs_serialization = any(isinstance(val, (list, tuple, dict)) for val in parsed_vals)
        if not needs_serialization:
            return lambda name: trial.suggest_categorical(name, parsed_vals)

        serialized_choices = []
        mapping = {}
        for val in parsed_vals:
            decoded = list(val) if isinstance(val, tuple) else val
            if isinstance(decoded, (list, dict)):
                serialized = json.dumps(decoded)
                serialized_choices.append(serialized)
                mapping[serialized] = decoded
            else:
                serialized_choices.append(decoded)
        def suggest(name):
            picked = trial.suggest_categorical(name, serialized_choices)
            return mapping.get(picked, picked)
        return suggest
    # Fallback: treat as categorical single value
    return lambda name: trial.suggest_categorical(name, [spec])


def build_space_from_params(trial: optuna.trial.Trial, params_node: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(params_node, dict) or not params_node:
        raise ValueError("No params found in sweeper config.")
    suggested: Dict[str, Any] = {}
    for key, spec in params_node.items():
        suggest_fn = _parse_sweeper_spec(str(spec), trial)
        val = suggest_fn(key)
        suggested[key] = val
    return suggested


def run_trial(cmd_base: list[str], run_dir: Path, skip_artifacts: bool = False,
              study_name: str | None = None, group: str = "hpo") -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"run.dir={str(run_dir)}",
        f"run.group={group}",
    ]
    if study_name:
        overrides.append(f"run.study={study_name}")
    full_cmd = cmd_base + overrides
    print("Launching:", " ".join(shlex.quote(p) for p in full_cmd))
    env = os.environ.copy()
    if skip_artifacts:
        env["RUN_TRIAL_SKIP_ARTIFACTS"] = "1"
        env.setdefault("RUN_TRIAL_SKIP_CHECKPOINTS", "1")
    proc = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    print(proc.stdout)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {}
    with results_path.open("r") as f:
        return json.load(f)


def _sorted_completed_trials(study, direction: str):
    try:
        completed = [
            t for t in study.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
    except Exception:
        return []
    reverse = direction == "maximize"
    return sorted(completed, key=lambda t: t.value, reverse=reverse)


def _best_completed_value_for_params(study, params: Dict[str, Any], direction: str):
    """Return best value among completed trials that match the exact params."""
    try:
        trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,), deepcopy=False)
    except Exception:
        return None
    best_val = None
    for t in trials:
        if t.params == params and t.value is not None:
            v = float(t.value)
            if best_val is None:
                best_val = v
            else:
                if direction == "maximize":
                    best_val = max(best_val, v)
                else:
                    best_val = min(best_val, v)
    return best_val


def _parse_trial_number(path: Path) -> int | None:
    name = path.name
    if not name.startswith("trial_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return None


def _prune_trial_dirs(trials_root: Path, keep_numbers: set[int], dry_run: bool = False):
    if not trials_root.exists():
        return
    for child in trials_root.iterdir():
        if not child.is_dir():
            continue
        num = _parse_trial_number(child)
        if num is None or num in keep_numbers:
            continue
        if dry_run:
            print(f"[hpo] DRY-RUN would delete trial dir: {child}")
            continue
        try:
            shutil.rmtree(child, ignore_errors=True)
        except Exception as e:
            print(f"[hpo] Failed to delete {child}: {e}")


def _params_to_overrides(params: Dict[str, Any]) -> list[str]:
    return [f"{k}={_format_override_value(v)}" for k, v in params.items()]


def _score_from_results(results: Dict[str, Any], metric: str):
    objective_info = (results or {}).get("objective", {}) or {}
    score = objective_info.get("best_score")
    if score is not None:
        try:
            return float(score)
        except Exception:
            pass
    fm = (results or {}).get("final_metrics", {}) or {}
    if metric == "val_f1":
        val = fm.get("val_f1_last") or fm.get("best_val_f1")
        if val is None:
            return None
        return float(val)
    else:
        val = fm.get("val_loss_last") or fm.get("best_val_loss")
        if val is None:
            return None
        return float(val)


def _write_topk_summary(trials: list, path: Path, metric: str, direction: str):
    entries = []
    for rank, t in enumerate(trials, start=1):
        entries.append(
            {
                "rank": rank,
                "trial_number": t.number,
                "value": t.value,
                "params": t.params,
            }
        )
    payload = {"metric": metric, "direction": direction, "trials": entries}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            if yaml:
                yaml.safe_dump(payload, f, sort_keys=False)
            else:
                json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[hpo] Failed to write topk summary: {e}")


def _format_choice(values):
    tokens = []
    for v in values:
        if isinstance(v, str):
            tokens.append(f"'{v}'")
        else:
            tokens.append(str(v))
    return f"choice({', '.join(tokens)})"


def _format_numeric(values):
    # Handle booleans explicitly (bool is subclass of int)
    if any(isinstance(v, bool) for v in values):
        counts = Counter(values)
        maxc = max(counts.values())
        top = [v for v, c in counts.items() if c == maxc]
        if len(top) == 1:
            return "choice(true)" if top[0] else "choice(false)"
        return f"choice({', '.join('true' if v else 'false' for v in top)})"

    vals = [float(v) for v in values]
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return lo if not lo.is_integer() else int(lo)
    if all(float(v).is_integer() for v in vals):
        return f"int({int(lo)}, {int(hi)})"
    if lo > 0 and hi > 0:
        return f"loguniform({lo}, {hi})"
    return f"uniform({lo}, {hi})"


def _write_topk_space(trials: list, path: Path, metric: str, direction: str):
    """
    Emit a suggested narrowed search space based on top-k trials.
    Numeric -> loguniform/uniform (ints -> int range); categorical/bool -> mode choice(s).
    """
    params_seen: Dict[str, list] = {}
    for t in trials:
        for k, v in t.params.items():
            params_seen.setdefault(k, []).append(v)

    suggested: Dict[str, str] = {}
    for k, vals in params_seen.items():
        numeric = True
        parsed = []
        for v in vals:
            try:
                parsed.append(float(v))
            except Exception:
                numeric = False
                break
        if numeric:
            suggested[k] = _format_numeric(parsed)
            continue
        counts = Counter(vals)
        maxc = max(counts.values())
        top_vals = [v for v, c in counts.items() if c == maxc]
        suggested[k] = _format_choice(top_vals)

    # Keep this minimal: only the refined sweeper params, no extra metadata.
    suggested_space = {"hydra": {"sweeper": {"params": suggested}}}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            if yaml:
                yaml.safe_dump(suggested_space, f, sort_keys=False)
            else:
                json.dump(suggested_space, f, indent=2)
    except Exception as e:
        print(f"[hpo] Failed to write topk_searchspace: {e}")


def _maintain_topk(
    study,
    out_root: Path,
    top_k: int,
    metric: str,
    direction: str,
    dry_run: bool,
    last_trial=None,
):
    if top_k <= 0:
        return
    completed_sorted = _sorted_completed_trials(study, direction)
    # Deduplicate by params: keep best trial per unique params
    unique: Dict[str, Any] = {}
    for t in completed_sorted:
        key = json.dumps(t.params, sort_keys=True)
        if key not in unique:
            unique[key] = t
    top_trials = list(unique.values())[:top_k] if unique else []
    if not top_trials:
        return
    topk_path = out_root / "topk.yaml"

    def _is_better(val, ref, mode):
        if ref is None:
            return True
        return val > ref if mode == "maximize" else val < ref

    # Decide whether to update based on last_trial vs current worst in existing topk
    should_update = False
    if topk_path.exists():
        try:
            with topk_path.open("r") as f:
                if yaml:
                    current = (yaml.safe_load(f) or {}).get("trials", [])
                else:
                    current = json.load(f).get("trials", [])
        except Exception:
            current = []
        current_vals = [t.get("value") for t in current if t.get("value") is not None]
        worst_current = None
        if current_vals:
            worst_current = min(current_vals) if direction == "maximize" else max(current_vals)
        if len(current) < top_k:
            should_update = True
        elif last_trial and getattr(last_trial, "value", None) is not None:
            should_update = _is_better(last_trial.value, worst_current, direction)
        else:
            should_update = True
    else:
        should_update = True

    if not should_update:
        return

    _write_topk_summary(top_trials, out_root / "topk.yaml", metric, direction)
    _write_topk_space(top_trials, out_root / "topk_searchspace.yaml", metric, direction)


def _extract_yaml_meta(yaml_path: Path, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Extract optional metadata from conf/hpo YAML."""
    meta: Dict[str, Any] = {
        "study_name": None,
        "trial": None,
        "metric": None,
        "mode": None,
        "trainer_overrides": [],
        "data_overrides": [],
        "top_k": 1,
        "repeat_enabled": False,
        "repeat_k": 1,
    }
    if yaml is None and data is None:
        return meta
    try:
        if data is None:
            with yaml_path.open("r") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = data or {}
        # Prefer explicit top-level study_name
        if isinstance(data.get("study_name"), str):
            meta["study_name"] = data.get("study_name")
        # hydra.sweeper.study_name
        sweeper = (data.get("hydra", {}) or {}).get("sweeper", {}) or {}
        if isinstance(sweeper, dict):
            if isinstance(sweeper.get("study_name"), str):
                # Avoid propagating unresolved templates like "${study_name}"
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
                objective = trainer_node.get("objective", {}) or {}
                if isinstance(objective, dict):
                    if isinstance(objective.get("metric"), str):
                        meta["metric"] = objective.get("metric")
                    if isinstance(objective.get("mode"), str):
                        meta["mode"] = objective.get("mode")

        data_node = data.get("data")
        if data_node is not None:
            if isinstance(data_node, str):
                data_node = {"name": data_node}
            if isinstance(data_node, dict):
                meta["data_overrides"] = _flatten_overrides("data", data_node)
        try:
            top_k = data.get("top_k", None)
            if isinstance(top_k, int) and top_k > 0:
                meta["top_k"] = top_k
        except Exception:
            pass
        repeat_node = data.get("repeat", {}) or {}
        if isinstance(repeat_node, dict):
            try:
                meta["repeat_enabled"] = bool(repeat_node.get("enabled", False))
            except Exception:
                pass
            try:
                k_val = repeat_node.get("count", repeat_node.get("k", None))
                if isinstance(k_val, int) and k_val > 0:
                    meta["repeat_k"] = k_val
            except Exception:
                pass
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
    parser.add_argument(
        "--search-mode",
        type=str,
        default="coarse",
        choices=["coarse", "fine"],
        help="Use the original search space (coarse) or a narrowed space from topk_searchspace (fine)",
    )
    parser.add_argument("--prune", action="store_true", help="Enable MedianPruner")
    parser.add_argument("--space-config", type=str, default=None, help="Hydra sweeper YAML defining the search space (conf/hpo/*.yaml)")
    parser.add_argument("--dry", action="store_true", help="Print commands without running")
    # Explicit run-time configuration (avoid free-form overrides for simplicity)
    parser.add_argument("--env", type=str, default="remote", help="Hydra env choice (e.g., local, remote)")
    parser.add_argument("--epochs", type=int, default=None, help="Trainer epochs override; if omitted, use config default")

    args = parser.parse_args()

    # Resolve space YAML first and extract metadata to drive study/metric.
    # If resuming an existing run (snapshots/hpo.yaml), always use that snapshot.
    # If a snapshot exists and space_mode=fine, error out (fine search should start as a new study).
    # Otherwise, use the provided space-config resolver.
    base_out_root = Path(args.output_root) if args.output_root else default_output_root(args.study_name)
    snapshot_space = base_out_root / "snapshots" / "hpo.yaml"
    snapshot_trial = base_out_root / "snapshots" / "trial.yaml"
    report_space = base_out_root / "topk_searchspace.yaml"
    using_snapshot = False
    study_root_exists = base_out_root.exists()
    # Only enforce snapshot presence if we truly need to resume (db/trials present).
    db_path = base_out_root / f"{args.study_name}.db"
    trials_path = base_out_root / "trials"
    resume_expected = snapshot_space.exists() or db_path.exists() or trials_path.exists()
    if resume_expected and (not snapshot_space.exists() or not snapshot_trial.exists()):
        missing = []
        if not snapshot_space.exists():
            missing.append("hpo.yaml")
        if not snapshot_trial.exists():
            missing.append("trial.yaml")
        raise FileNotFoundError(
            f"Study '{args.study_name}' exists at {base_out_root} but snapshots missing: {', '.join(missing)}; cannot resume."
        )

    if snapshot_space.exists():
        if args.search_mode == "fine" and report_space.exists():
            raise ValueError(
                f"Study '{args.study_name}' already exists with snapshots; fine mode must start a new study or RUN_NAME."
            )
        space_yaml_path = snapshot_space
        using_snapshot = True
    else:
        space_yaml_path = _resolve_space_yaml(args)
    space_yaml_data = None
    if yaml is None:
        raise RuntimeError("PyYAML not available; cannot load --space-config")
    with space_yaml_path.open("r") as f:
        space_yaml_data = yaml.safe_load(f) or {}
    # Snapshot the HPO YAML and trial YAML for reproducibility/resume
    snapshot_dir = base_out_root / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if not using_snapshot:
        try:
            (snapshot_dir / "hpo.yaml").write_text(space_yaml_path.read_text())
        except Exception:
            pass
    trial_name_for_snapshot = None
    snapshot_trial_path = snapshot_dir / "trial.yaml"
    try:
        trial_name_for_snapshot = space_yaml_data.get("trial")
    except Exception:
        trial_name_for_snapshot = None
    if trial_name_for_snapshot and not using_snapshot:
        trial_yaml_path = Path("conf/trial") / f"{trial_name_for_snapshot}.yaml"
        if trial_yaml_path.exists():
            try:
                snapshot_trial_path.write_text(trial_yaml_path.read_text())
            except Exception:
                pass
    meta = _extract_yaml_meta(space_yaml_path, space_yaml_data)

    trainer_overrides = meta.get("trainer_overrides") or []
    data_overrides = meta.get("data_overrides") or []
    top_k = max(1, int(meta.get("top_k") or 1))
    repeat_enabled = bool(meta.get("repeat_enabled"))
    repeat_k = max(1, int(meta.get("repeat_k") or 1))

    # Adopt study name from YAML if provided, unless user explicitly passed one
    default_study = parser.get_default("study_name")
    if args.study_name != default_study:
        study_name = args.study_name
    else:
        study_name = meta.get("study_name") or args.study_name

    # Determine metric and direction; prefer trainer objective if present
    metric = meta.get("metric") or args.metric
    if args.direction is None:
        direction = _mode_to_direction(meta.get("mode"))
        if direction is None:
            direction = "maximize" if metric == "val_f1" else "minimize"
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
    # Build explicit CLI overrides from flags
    base_overrides = [
        f"env={args.env}",
        f"seed={args.seed}",
    ]
    if trainer_overrides:
        base_overrides.extend(trainer_overrides)
    if data_overrides:
        base_overrides.extend(data_overrides)
    trial_override = meta.get("trial")
    if trial_override:
        base_overrides.append(f"trial={trial_override}")
    if args.epochs is not None:
        base_overrides.append(f"trainer.epochs={args.epochs}")

    # Default epochs from trial config (used for repeat stage to revert to canonical value)
    default_trial_epochs = _load_trial_default_epochs(trial_override)

    # Extract sweeper params once to avoid mid-run mutations
    try:
        params_node = (space_yaml_data.get("hydra", {}) or {}).get("sweeper", {}) or {}
        params_node = params_node.get("params", {}) if isinstance(params_node, dict) else {}
    except Exception:
        params_node = {}
    if not isinstance(params_node, dict) or not params_node:
        raise ValueError(f"No params found in YAML sweeper file: {space_yaml_path}")

    # YAML path already resolved above; fail-fast happened earlier

    def objective(trial: optuna.trial.Trial) -> float:
        params = build_space_from_params(trial, params_node)

        # Skip exact duplicate parameter sets to save compute; reuse best completed value.
        dup_val = _best_completed_value_for_params(study, params, direction)
        if dup_val is not None:
            print(f"[hpo] Duplicate params detected (trial {trial.number}); reusing value {dup_val:.4f}")
            return dup_val

        trial_dir = out_root / "trials" / f"trial_{trial.number:04d}"

        # Convert params dict to hydra-style overrides
        trial_overrides = base_overrides + [f"{k}={v}" for k, v in params.items()]
        cmd = base_cmd + trial_overrides

        if args.dry:
            print("DRY RUN:", " ".join(shlex.quote(x) for x in cmd))
            return 0.0

        results = run_trial(cmd, trial_dir, skip_artifacts=True, study_name=study_name)
        if not results:
            # Failed run; assign a bad score so it's pruned from consideration
            return -1e9 if direction == "maximize" else 1e9

        score = _score_from_results(results, metric)
        if score is None:
            return -1e9 if direction == "maximize" else 1e9
        return score

    def _trial_callback(study, trial):
        _maintain_topk(study, out_root, top_k, metric, direction, args.dry, last_trial=trial)

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True, callbacks=[_trial_callback])

    completed_sorted = _sorted_completed_trials(study, direction)
    # Deduplicate by params to avoid re-pruning the same config and causing mismatches
    unique: Dict[str, Any] = {}
    for t in completed_sorted:
        key = json.dumps(t.params, sort_keys=True)
        if key not in unique:
            unique[key] = t
    top_trials = list(unique.values())[:top_k] if top_k > 0 else []
    if top_trials:
        keep_numbers = {t.number for t in top_trials}
        _prune_trial_dirs(out_root / "trials", keep_numbers, dry_run=args.dry)
        _write_topk_summary(top_trials, out_root / "topk.yaml", metric, direction)
        _write_topk_space(top_trials, out_root / "topk_searchspace.yaml", metric, direction)

    repeat_summary = []
    if repeat_enabled and top_trials and args.search_mode == "fine":
        repeats_root = out_root / "repeats"
        for rank, t in enumerate(top_trials, start=1):
            params_overrides = _params_to_overrides(t.params)
            # For repeat stage, drop sweep-specific overrides that shouldn't persist (epochs/seed) and restore defaults
            repeat_base_overrides = [
                ov for ov in base_overrides if not (ov.startswith("trainer.epochs=") or ov.startswith("seed="))
            ]
            if default_trial_epochs:
                repeat_base_overrides.append(f"trainer.epochs={default_trial_epochs}")
            for rep_idx in range(repeat_k):
                rep_dir = repeats_root / f"trial_{t.number:04d}_rep_{rep_idx}"
                repeat_seed = args.seed + t.number * repeat_k + rep_idx
                rep_cmd = base_cmd + repeat_base_overrides + params_overrides + [f"seed={repeat_seed}"]
                if args.dry:
                    print(
                        f"DRY RUN repeat (rank {rank}, trial {t.number}, rep {rep_idx}):",
                        " ".join(shlex.quote(x) for x in rep_cmd),
                    )
                    continue
                rep_results = run_trial(rep_cmd, rep_dir, skip_artifacts=True, study_name=study_name, group="repeat")
                rep_score = _score_from_results(rep_results, metric) if rep_results else None
                repeat_summary.append(
                    {
                        "rank": rank,
                        "trial_number": t.number,
                        "repeat_index": rep_idx,
                        "value": rep_score,
                        "run_dir": str(rep_dir),
                    }
                )
        if repeat_summary and not args.dry:
            try:
                repeats_root.mkdir(parents=True, exist_ok=True)
                # Build repeats report (mean/std per trial_number)
                grouped: Dict[int, Dict[str, Any]] = {}
                for entry in repeat_summary:
                    tnum = entry["trial_number"]
                    grouped.setdefault(tnum, {"values": []})
                    if entry.get("value") is not None:
                        grouped[tnum]["values"].append(entry["value"])
                repeat_report = {"metric": metric, "direction": direction, "trials": []}
                params_lookup = {t.number: t.params for t in top_trials}
                for tnum, info in grouped.items():
                    vals = info["values"]
                    if not vals:
                        continue
                    avg = sum(vals) / len(vals)
                    var = sum((v - avg) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
                    entry = {
                        "trial_number": tnum,
                        "mean": avg,
                        "std": var ** 0.5,
                        "count": len(vals),
                    }
                    repeat_report["trials"].append(entry)
                reverse = direction == "maximize"
                repeat_report["trials"].sort(key=lambda x: x["mean"], reverse=reverse)
                best_params = None
                if repeat_report["trials"]:
                    best_tnum = repeat_report["trials"][0]["trial_number"]
                    best_params = params_lookup.get(best_tnum, {})
                with (out_root / "repeats_report.yaml").open("w") as f:
                    if yaml:
                        yaml.safe_dump(repeat_report, f, sort_keys=False)
                    else:
                        json.dump(repeat_report, f, indent=2)
                if best_params is not None:
                    with (out_root / "repeats_best_params.yaml").open("w") as f:
                        if yaml:
                            yaml.safe_dump(best_params, f, sort_keys=False)
                        else:
                            json.dump(best_params, f, indent=2)
            except Exception as e:
                print(f"[hpo] Failed to write repeats summary: {e}")

    if study.best_trial:
        summary = {
            "best_value": study.best_trial.value,
            "best_params": study.best_trial.params,
            "best_trial_number": study.best_trial.number,
            "direction": direction,
            "metric": metric,
            "storage": storage_url,
            "study_name": study_name,
        }
        print("HPO complete. Best:", json.dumps(summary, indent=2))
    else:
        print("HPO complete. No successful trials.")


if __name__ == "__main__":
    main()
