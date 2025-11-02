#!/usr/bin/env python
"""Single-trial Experiment Runner (Hydra-based)

Usage examples:
    python experiments/run_trial.py data=mmfit_debug trainer.epochs=2
    python experiments/run_trial.py scenario=scenario2 trainer.epochs=5 optim.lr=5e-4

Module usage:
    python -m experiments.run_trial data=mmfit trainer.epochs=10
"""
from .run import main

if __name__ == "__main__":
    main()
