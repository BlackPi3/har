#!/usr/bin/env python
"""Compose and materialize Hydra config for reproducibility.

Usage:
  python tests/test_conf.py                     # default config
  python tests/test_conf.py env=remote data=utd

Also works via module:
  python -m tests.test_conf env=remote data=utd

Outputs:
  Prints concise summary to stdout.
  Writes full resolved config to: test_outputs/resolved_config.yaml and .json
  Exits with code 0 if all required sections present.

This does NOT run training; it's only a configuration integrity check.
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import hydra
from omegaconf import OmegaConf, DictConfig

REQUIRED_SECTIONS = [
    'env', 'experiment', 'data', 'model', 'optim', 'trainer'
]


@hydra.main(config_path="../conf", config_name="conf", version_base=None)
def compose_only(cfg: DictConfig):
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        print(f"ERROR: Missing required config sections: {missing}")
        sys.exit(1)

    summary = {
        'env': {k: cfg.env.get(k) for k in ['device', 'data_dir'] if k in cfg.env},
        'experiment': {k: cfg.experiment.get(k) for k in ['experiment_name', 'alpha', 'beta'] if k in cfg.experiment},
        'data': {k: cfg.data.get(k) for k in ['name', 'batch_size', 'num_workers'] if k in cfg.data},
        'optim': {k: cfg.optim.get(k) for k in ['name', 'lr', 'weight_decay'] if k in cfg.optim},
        'trainer': {k: cfg.trainer.get(k) for k in ['epochs', 'patience'] if k in cfg.trainer},
        'model_keys': list(cfg.model.keys()) if 'model' in cfg else [],
    }

    orig_root = Path(hydra.utils.get_original_cwd())
    out_dir = orig_root / 'tests/test_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    with (out_dir / 'resolved_config.yaml').open('w') as f_yaml:
        f_yaml.write(OmegaConf.to_yaml(cfg, resolve=True))
    with (out_dir / 'resolved_config.json').open('w') as f_json:
        json.dump(resolved, f_json, indent=2)

    print("Hydra config composition successful. Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Full resolved config written to {out_dir}/resolved_config.*")
    import hashlib
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()[:16]
    print(f"Config content hash: {digest}")


if __name__ == '__main__':
    compose_only()
