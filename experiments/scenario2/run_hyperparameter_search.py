#!/usr/bin/env python3
"""
Hyperparameter Search Runner for HAR Experiments

Usage:
    python run_hyperparameter_search.py --search-config coarse_search.yaml --trial-id 0
    python run_hyperparameter_search.py --search-config fine_search.yaml --trial-id 5
"""

import sys
import os
import argparse
import yaml
import json
import itertools
import random
from pathlib import Path
from datetime import datetime
import subprocess

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import after path setup
from src.config import load_config
from src.data import get_dataloaders
from src.train import Trainer
from src.models import Regressor, FeatureExtractor, ActivityClassifier


def load_search_config(search_config_path):
    """Load hyperparameter search configuration."""
    with open(search_config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_parameter_combinations(search_space, search_method, n_trials):
    """Generate parameter combinations based on search method."""
    
    if search_method == "grid":
        # Generate all combinations (grid search)
        keys = search_space.keys()
        values = search_space.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]
    
    elif search_method == "random":
        # Generate random combinations
        combinations = []
        for _ in range(n_trials):
            combo = {}
            for key, values in search_space.items():
                combo[key] = random.choice(values)
            combinations.append(combo)
        return combinations
    
    elif search_method == "bayesian":
        # TODO: Implement Optuna integration
        raise NotImplementedError("Bayesian optimization not implemented yet")
    
    else:
        raise ValueError(f"Unknown search method: {search_method}")


def apply_hyperparameters(base_cfg, hyperparams):
    """Apply hyperparameters to configuration."""
    # Simple implementation - extend as needed
    for key, value in hyperparams.items():
        if key == 'lr':
            base_cfg.lr = value
        elif key == 'alpha':
            base_cfg.alpha = value
        elif key == 'models':
            # Handle nested model parameters
            for model_name, model_params in value.items():
                if hasattr(base_cfg.models, model_name):
                    model_cfg = getattr(base_cfg.models, model_name)
                    for param_name, param_value in model_params.items():
                        setattr(model_cfg, param_name, param_value)
    
    return base_cfg


def run_single_trial(search_cfg, hyperparams, trial_id, output_dir):
    """Run a single hyperparameter trial."""
    
    print(f"=== Trial {trial_id} ===")
    print(f"Hyperparameters: {hyperparams}")
    
    try:
        # Load base configuration
        base_cfg = load_config(
            search_cfg['base_config'], 
            search_cfg['experiment_config']
        )
        
        # Apply hyperparameters
        cfg = apply_hyperparameters(base_cfg, hyperparams)
        
        # Override training settings from search config
        if 'training' in search_cfg:
            training_cfg = search_cfg['training']
            cfg.epochs = training_cfg.get('epochs', cfg.epochs)
            cfg.patience = training_cfg.get('patience', cfg.patience)
        
        # Create dataloaders
        dls = get_dataloaders(cfg.dataset_name, cfg)
        
        # Build models
        device = getattr(cfg, 'torch_device', torch.device(cfg.device))
        models = {
            'pose2imu': Regressor(
                in_ch=cfg.models.regressor.input_channels,
                num_joints=cfg.models.regressor.num_joints,
                window_length=cfg.models.regressor.sequence_length,
            ).to(device),
            'fe': FeatureExtractor().to(device),
            'ac': ActivityClassifier(
                f_in=cfg.models.classifier.f_in, 
                n_classes=cfg.models.classifier.n_classes
            ).to(device),
        }
        
        # Setup training
        lr = float(cfg.lr) if isinstance(cfg.lr, str) else cfg.lr
        params = sum([list(m.parameters()) for m in models.values()], [])
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=cfg.patience
        )
        
        trainer = Trainer(
            models=models, 
            dataloaders=dls, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            cfg=cfg, 
            device=device
        )
        
        # Run training
        start_time = datetime.now()
        history = trainer.fit(cfg.epochs)
        end_time = datetime.now()
        
        # Calculate metrics
        train_time = (end_time - start_time).total_seconds()
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
        best_epoch = min(enumerate(history['val_loss']), key=lambda x: x[1])[0] if history['val_loss'] else 0
        
        # Save results
        results = {
            'trial_id': trial_id,
            'hyperparameters': hyperparams,
            'metrics': {
                'val_loss': final_val_loss,
                'train_time_seconds': train_time,
                'best_epoch': best_epoch,
                'converged': len(history['val_loss']) < cfg.epochs  # Early stopping
            },
            'history': history,
            'config': {k: str(v) for k, v in vars(cfg).items()},
            'timestamp': datetime.now().isoformat(),
            'git_commit': get_git_commit()
        }
        
        # Save to file
        results_file = output_dir / f"trial_{trial_id:03d}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Trial {trial_id} completed: val_loss={final_val_loss:.4f}, time={train_time:.1f}s")
        return results
        
    except Exception as e:
        print(f"✗ Trial {trial_id} failed: {e}")
        # Save failure info
        failure_info = {
            'trial_id': trial_id,
            'hyperparameters': hyperparams,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        failure_file = output_dir / f"trial_{trial_id:03d}_FAILED.json"
        with open(failure_file, 'w') as f:
            json.dump(failure_info, f, indent=2)
        
        return None


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], 
            capture_output=True, 
            text=True, 
            cwd=project_root
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter search')
    parser.add_argument('--search-config', required=True, 
                       help='Path to search configuration file')
    parser.add_argument('--trial-id', type=int, required=True,
                       help='Trial ID for this run')
    parser.add_argument('--output-dir', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load search configuration
    search_cfg = load_search_config(args.search_config)
    
    # Generate parameter combinations
    # Extract simple search space for now (extend for nested parameters)
    simple_search_space = {
        k: v for k, v in search_cfg['search_space'].items() 
        if k not in ['models', 'optimizer']
    }
    
    combinations = generate_parameter_combinations(
        simple_search_space,
        search_cfg['search_method'],
        search_cfg['n_trials']
    )
    
    if args.trial_id >= len(combinations):
        print(f"Trial ID {args.trial_id} out of range (max: {len(combinations)-1})")
        return
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        search_name = Path(args.search_config).stem
        output_dir = Path(f"outputs/hyperparameter_search/{search_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the specific trial
    hyperparams = combinations[args.trial_id]
    results = run_single_trial(search_cfg, hyperparams, args.trial_id, output_dir)
    
    if results:
        print(f"Results saved to: {output_dir}")
    else:
        print(f"Trial failed - check logs in: {output_dir}")


if __name__ == "__main__":
    import torch  # Import here to avoid issues in argument parsing
    main()