from types import SimpleNamespace
from pathlib import Path
import yaml
import json
import os
import random
import torch
import numpy as np


def _dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    return d


def merge_dicts(a: dict, b: dict) -> dict:
    """Shallow merge of two dicts with nested dict support (b overrides a)."""
    out = dict(a) if a else {}
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def load_config(base_path: str = None, exp_path: str = None, opts: list = None):
    """Load base + experiment YAML, apply simple overrides and normalize device.

    Args:
        base_path: path to base YAML (optional)
        exp_path: path to experiment YAML (optional)
        opts: list of override strings like ["a=1", "nested.key=true"]

    Returns:
        SimpleNamespace with merged config and helpers: .device (str), .torch_device, .cluster, .dtype, .np_dtype
    """
    base = {}
    exp = {}
    if base_path:
        p = Path(base_path)
        if p.exists():
            base = yaml.safe_load(p.read_text()) or {}
    if exp_path:
        q = Path(exp_path)
        if q.exists():
            exp = yaml.safe_load(q.read_text()) or {}

    cfg_dict = merge_dicts(base, exp)

    # Apply simple CLI overrides: list of 'key.path=value'
    for o in opts or []:
        if not isinstance(o, str) or "=" not in o:
            continue
        k, v = o.split("=", 1)
        try:
            val = json.loads(v)
        except Exception:
            val = v
        parts = k.split(".")
        d = cfg_dict
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = val

    # Detect cluster by SLURM env vars
    is_cluster = bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_ON_NODE"))

    # If running on cluster and cluster_overrides are defined, merge them (b overrides a)
    if is_cluster and isinstance(cfg_dict.get("cluster_overrides"), dict):
        overrides = cfg_dict.pop("cluster_overrides")  # remove to avoid persisting raw section
        cfg_dict = merge_dicts(cfg_dict, overrides)
        cfg_dict["cluster_overrides_applied"] = True

    # Device selection
    if is_cluster:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
        if mps_available:
            device_str = "mps"
        elif torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"

    # Respect explicit device in YAML
    explicit = cfg_dict.get("device")
    if explicit not in (None, "", "auto"):
        device_str = explicit

    cfg_dict["device"] = device_str
    cfg = _dict_to_ns(cfg_dict)
    try:
        cfg.torch_device = torch.device(cfg.device)
    except Exception:
        # fallback to cpu
        cfg.torch_device = torch.device("cpu")
        cfg.device = "cpu"
    cfg.cluster = is_cluster
    cfg.dtype = torch.float32
    cfg.np_dtype = np.float32
    return cfg


def set_seed(seed: int):
    """Set random seeds for python, numpy and torch (CUDA if available)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)