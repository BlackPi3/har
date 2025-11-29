"""
Factory function for creating MMFit datasets.
"""
import os
import torch
from torch.utils.data import ConcatDataset
from src.config import get_data_cfg_value
from .dataset import MMFit
from .constants import (
    DEFAULT_TRAIN_SUBJECTS, DEFAULT_VAL_SUBJECTS, DEFAULT_TEST_SUBJECTS,
    TRAIN_SIM_SUBJECTS, DEFAULT_POSE_FILE, DEFAULT_ACC_FILE, 
    DEFAULT_SIM_ACC_FILE, DEFAULT_LABELS_FILE, DEFAULT_SENSOR_WINDOW_LENGTH
)


def build_mmfit_datasets(cfg):
    """
    Factory function to build MMFit train/val/test datasets based on config.
    This function is called by src.data.get_dataloaders().
    
    Args:
        cfg: Configuration object with mmfit-specific attributes
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) as ConcatDataset objects
    """
    # Extract configuration values with fallbacks
    data_dir = get_data_cfg_value(cfg, 'data_dir', '../datasets/mm-fit/')
    train_ids = get_data_cfg_value(cfg, 'train_subjects', DEFAULT_TRAIN_SUBJECTS)
    val_ids = get_data_cfg_value(cfg, 'val_subjects', DEFAULT_VAL_SUBJECTS)
    test_ids = get_data_cfg_value(cfg, 'test_subjects', DEFAULT_TEST_SUBJECTS)
    
    pose_file = get_data_cfg_value(cfg, 'pose_file', DEFAULT_POSE_FILE)
    acc_file = get_data_cfg_value(cfg, 'acc_file', DEFAULT_ACC_FILE)
    labels_file = get_data_cfg_value(cfg, 'labels_file', DEFAULT_LABELS_FILE)
    sim_acc_file = get_data_cfg_value(cfg, 'sim_acc_file', DEFAULT_SIM_ACC_FILE)
    
    sensor_window_length = int(get_data_cfg_value(cfg, 'sensor_window_length', DEFAULT_SENSOR_WINDOW_LENGTH))
    stride_seconds = get_data_cfg_value(cfg, 'stride_seconds', None)
    sampling_rate_hz = int(get_data_cfg_value(cfg, 'sampling_rate_hz', 100))
    
    use_simulated_data = bool(get_data_cfg_value(cfg, 'use_simulated_data', False))
    
    train, val, test = [], [], []
    
    for w_id in train_ids + val_ids + test_ids:
        id_dir = os.path.join(data_dir, w_id)
        pose_file_path = os.path.join(id_dir, f"{w_id}_{pose_file}")
        acc_file_path = os.path.join(id_dir, f"{w_id}_{acc_file}")
        labels_file_path = os.path.join(id_dir, f"{w_id}_{labels_file}")

        # Use simulated accelerometer data for specific subjects if configured
        if use_simulated_data and w_id in TRAIN_SIM_SUBJECTS:
            acc_file_path = os.path.join(id_dir, f"{w_id}_{sim_acc_file}")

        try:
            dataset = MMFit(
                pose_file=pose_file_path,
                acc_file=acc_file_path,
                labels_file=labels_file_path,
                sensor_window_length=sensor_window_length,
                stride_seconds=stride_seconds,
                sampling_rate_hz=sampling_rate_hz,
                cluster=getattr(cfg, 'cluster', False),
                dtype=getattr(cfg, 'dtype', None) or torch.float32,
            )
            
            # Add to appropriate split
            if w_id in train_ids:
                train.append(dataset)
            elif w_id in val_ids:
                val.append(dataset)
            elif w_id in test_ids:
                test.append(dataset)
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load dataset for subject {w_id}: {e}")
            continue

    # Return ConcatDatasets (no artificial debug limiting here)
    train_dataset = ConcatDataset(train) if train else None
    val_dataset = ConcatDataset(val) if val else None
    test_dataset = ConcatDataset(test) if test else None
    
    return train_dataset, val_dataset, test_dataset
