import argparse
import importlib
import yaml
from pathlib import Path
import time
import json
from src.config import load_config, set_seed

def _ns_to_dict(ns):
    if isinstance(ns, dict):
        return ns
    d = {}
    for k, v in vars(ns).items():
        if hasattr(v, "__dict__"):
            d[k] = _ns_to_dict(v)
        else:
            d[k] = v
    return d

def save_merged_config(cfg, experiment_name, out_root="outputs"):
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_root) / experiment_name / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = _ns_to_dict(cfg)
    with open(out_dir / "config_merged.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    # optional: also save json for easy programmatic parsing
    with open(out_dir / "config_merged.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    return out_dir

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, help="module name in src.experiments (e.g. scenario2)")
    p.add_argument("--config", required=True, help="path to experiment yaml (e.g. configs/scenario2.yaml)")
    p.add_argument("--base-config", default="configs/base.yaml", help="base defaults yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--opts", nargs="*", default=[])
    args = p.parse_args()

    cfg = load_config(args.base_config, args.config, opts=args.opts)
    if args.seed is not None:
        cfg.seed = args.seed
    set_seed(getattr(cfg, "seed", None))

    module_path = f"src.experiments.{args.experiment}"
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "main"):
        raise SystemExit(f"{module_path} missing main(cfg)")
    print(f"Running {module_path} with config {args.config}")
    mod.main(cfg)

if __name__ == "__main__":
    main()