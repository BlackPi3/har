#!/usr/bin/env python3
"""
Quick test for cluster setup
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

print("=== Testing Scenario 2 Setup ===")
print(f"Project root: {project_root}")

try:
    from src.config import load_config
    print("✓ Config module imported")
    
    cfg = load_config('../../configs/base.yaml', 'configs/scenario2.yaml', opts=['epochs', '1'])
    print("✓ Configuration loaded")
    
    # Fix relative paths
    if hasattr(cfg, 'data_dir') and cfg.data_dir.startswith('../'):
        project_root_path = Path(__file__).parent.parent.parent
        cfg.data_dir = str((project_root_path / cfg.data_dir.lstrip('../')).resolve())
    
    print(f"  Device: {cfg.device}")
    print(f"  Dataset: {cfg.dataset_name}")
    print(f"  Data dir: {cfg.data_dir}")
    print(f"  Data dir exists: {os.path.exists(cfg.data_dir)}")
    
    from src.data import get_dataloaders
    print("✓ Data module imported")
    
    # Try to load just one subject to test
    original_train = cfg.train_subjects
    cfg.train_subjects = ['w00']  # Just one subject for testing
    cfg.val_subjects = ['w01'] if 'w01' in original_train else ['w00']
    cfg.test_subjects = ['w02'] if 'w02' in original_train else ['w00']
    
    dls = get_dataloaders(cfg.dataset_name, cfg)
    print("✓ Dataloaders created")
    print(f"  Dataset sizes: {[(k, len(v.dataset) if v.dataset is not None else 'None') for k, v in dls.items()]}")
    
    from src.models import Regressor, FeatureExtractor, ActivityClassifier
    print("✓ Models imported")
    
    print("\n=== Setup Complete - Ready for cluster! ===")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
