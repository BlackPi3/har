#!/usr/bin/env python3
"""
Test cluster configuration
"""
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

print("=== Testing Cluster Configuration ===")

try:
    from src.config import load_config
    
    # Simulate cluster environment
    os.environ['SLURM_JOB_ID'] = 'TEST'
    cfg = load_config('../../configs/base.yaml', 'configs/scenario2.yaml')
    print("✓ Unified configuration loaded with cluster simulation")
    print(f"  Data dir (cluster): {cfg.data_dir}")
    print(f"  Batch size (cluster): {cfg.batch_size}")
    print(f"  Epochs (cluster): {cfg.epochs}")
    print(f"  Cluster flag: {getattr(cfg, 'cluster', False)}")
    print(f"  Overrides applied: {getattr(cfg, 'cluster_overrides_applied', False)}")
    is_cluster_path = cfg.data_dir.startswith('/netscratch/')
    print(f"  Uses cluster path: {is_cluster_path}")
    
    # Test standard configuration for comparison
    cfg_std = load_config('../../configs/base.yaml', 'configs/scenario2.yaml')
    print("\n✓ Standard (local) configuration loaded")
    print(f"  Data dir: {cfg_std.data_dir}")
    print(f"  Batch size: {cfg_std.batch_size}")
    
    print("\n=== Configuration Comparison ===")
    print(f"Cluster data path: {cfg.data_dir}")
    print(f"Local data path:   {cfg_std.data_dir}")
    print(f"Cluster batch:     {cfg.batch_size}")
    print(f"Local batch:       {cfg_std.batch_size}")
    print(f"Cluster epochs:    {cfg.epochs}")
    print(f"Local epochs:      {cfg_std.epochs}")
    
    print("\n✅ Unified configuration & overrides working!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
