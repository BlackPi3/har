#!/usr/bin/env python3
"""DEPRECATED: please use `experiments/run_optuna.py` instead.

This legacy wrapper forwards all arguments to `run_optuna.py` so that
existing scripts continue to work.
"""
import sys
from pathlib import Path
import subprocess


def main() -> int:
    here = Path(__file__).resolve()
    run_optuna = here.with_name("run_optuna.py")
    print("[DEPRECATION] experiments/run_hpo.py is renamed to experiments/run_optuna.py. Forwarding...", file=sys.stderr)
    cmd = [sys.executable, str(run_optuna)] + sys.argv[1:]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
