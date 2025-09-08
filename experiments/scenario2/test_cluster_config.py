#!/usr/bin/env python3
"""
Test cluster configuration
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

print("=== Testing Cluster Configuration ===")

try:
    from src.config import load_config
    
    # Test cluster configuration
    cfg = load_config('../../configs/base.yaml', 'configs/scenario2_cluster.yaml')
    print("✓ Cluster configuration loaded")
    print(f"  Data dir: {cfg.data_dir}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Cluster flag: {getattr(cfg, 'cluster', False)}")
    
    # Check if this looks like cluster path
    is_cluster_path = cfg.data_dir.startswith('/netscratch/')
    print(f"  Uses cluster path: {is_cluster_path}")
    
    # Test standard configuration for comparison
    cfg_std = load_config('../../configs/base.yaml', 'configs/scenario2.yaml')
    print("\n✓ Standard configuration loaded")
    print(f"  Data dir: {cfg_std.data_dir}")
    print(f"  Batch size: {cfg_std.batch_size}")
    
    print(f"\n=== Configuration Comparison ===")
    print(f"Cluster data path: {cfg.data_dir}")
    print(f"Local data path:   {cfg_std.data_dir}")
    print(f"Cluster batch:     {cfg.batch_size}")
    print(f"Local batch:       {cfg_std.batch_size}")
    
    print(f"\n✅ Cluster configuration ready!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
