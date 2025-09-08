#!/usr/bin/env python3
"""
Scenario 2 Experiment Runner
"""
import sys
import os
import argparse
import torch
import time
import json
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.config import load_config, set_seed
from src.data import get_dataloaders
from src.train import Trainer
from src.models import Regressor, FeatureExtractor, ActivityClassifier


def run_scenario2_experiment(base_config=None, exp_config=None, opts=None, output_dir=None):
    """Run scenario 2 experiment with given configuration."""
    
    # Default paths relative to this script
    script_dir = Path(__file__).parent
    base_config = base_config or str(script_dir / "../../configs/base.yaml")
    exp_config = exp_config or str(script_dir / "../../configs/scenario2.yaml")
    
    # Load configuration
    cfg = load_config(base_config, exp_config, opts=opts or [])
    set_seed(getattr(cfg, 'seed', None))
    
    # Verify data directory exists
    if not os.path.exists(cfg.data_dir):
        print(f"Warning: Data directory not found: {cfg.data_dir}")
        print("Please ensure data is available at the specified path")
        if cfg.data_dir.startswith('./'):
            print("Local path detected - ensure you're running from project root")
        elif cfg.data_dir.startswith('/netscratch/'):
            print("Cluster path detected - ensure data is mounted/available")
    else:
        print(f"✓ Data directory found: {cfg.data_dir}")
    
    print(f"Running Scenario 2 Experiment")
    print(f"Base config: {base_config}")
    print(f"Experiment config: {exp_config}")
    print(f"Device: {cfg.device}")
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Data directory: {cfg.data_dir}")
    
    # Create output directory
    if output_dir is None:
        ts = time.strftime('%Y%m%d-%H%M%S')
        output_dir = script_dir / "outputs" / ts
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Create dataloaders
    dls = get_dataloaders(cfg.dataset_name, cfg)
    print('Dataset sizes:', {k: len(v.dataset) for k, v in dls.items()})
    
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
    
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print('Parameter counts:', {k: count_params(m) for k, m in models.items()})
    
    # Setup training
    lr = float(cfg.lr) if isinstance(cfg.lr, str) else cfg.lr
    params = sum([list(m.parameters()) for m in models.values()], [])
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cfg.patience)
    
    trainer = Trainer(models=models, dataloaders=dls, optimizer=optimizer, scheduler=scheduler, cfg=cfg, device=device)
    
    # Run training
    print(f"\nStarting training for {cfg.epochs} epochs...")
    history = trainer.fit(cfg.epochs)
    
    # Save results
    results = {
        'config': {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                  for k, v in vars(cfg).items()},
        'history': history,
        'final_metrics': {
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment completed!")
    print(f"results saved to: {output_dir}")
    
    return history, models


def main():
    parser = argparse.ArgumentParser(description='Run Scenario 2 Experiment')
    parser.add_argument('--base-config', type=str, help='Path to base config file')
    parser.add_argument('--exp-config', type=str, help='Path to experiment config file') 
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    # Build opts list from command line arguments
    opts = []
    if args.epochs is not None:
        opts.extend(['epochs', str(args.epochs)])
    if args.lr is not None:
        opts.extend(['lr', str(args.lr)])
    if args.seed is not None:
        opts.extend(['seed', str(args.seed)])
    
    run_scenario2_experiment(
        base_config=args.base_config,
        exp_config=args.exp_config,
        opts=opts,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()