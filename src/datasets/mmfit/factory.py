"""
Factory function for creating MMFit datasets.
"""
import os
import torch
from torch.utils.data import ConcatDataset
from src.config import get_data_cfg_value
from .dataset import MMFit
from .constants import TRAIN_SIM_SUBJECTS


def build_mmfit_datasets(cfg):
    """
    Factory function to build MMFit train/val/test datasets based on config.
    This function is called by src.data.get_dataloaders().
    
    Args:
        cfg: Configuration object with mmfit-specific attributes
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) as ConcatDataset objects
    """
    # Extract configuration values - no fallbacks for critical paths
    data_dir = get_data_cfg_value(cfg, 'data_dir', None)
    if data_dir is None:
        raise ValueError("data_dir is required for MMFit dataset. Set it in conf/data/mmfit.yaml or conf/env/*.yaml")
    
    train_ids = get_data_cfg_value(cfg, 'train_subjects', None)
    val_ids = get_data_cfg_value(cfg, 'val_subjects', None)
    test_ids = get_data_cfg_value(cfg, 'test_subjects', None)
    if train_ids is None:
        raise ValueError("train_subjects is required for MMFit dataset")
    if val_ids is None:
        val_ids = []
    if test_ids is None:
        test_ids = []
    
    pose_file = get_data_cfg_value(cfg, 'pose_file', None)
    if pose_file is None:
        raise ValueError("pose_file is required for MMFit dataset")
    acc_file = get_data_cfg_value(cfg, 'acc_file', None)
    if acc_file is None:
        raise ValueError("acc_file is required for MMFit dataset")
    labels_file = get_data_cfg_value(cfg, 'labels_file', None)
    if labels_file is None:
        raise ValueError("labels_file is required for MMFit dataset")
    sim_acc_file = get_data_cfg_value(cfg, 'sim_acc_file', None)  # optional
    
    sensor_window_length = get_data_cfg_value(cfg, 'sensor_window_length', None)
    if sensor_window_length is None:
        raise ValueError("sensor_window_length is required for MMFit dataset")
    sensor_window_length = int(sensor_window_length)
    
    stride_seconds = get_data_cfg_value(cfg, 'stride_seconds', None)
    sampling_rate_hz = get_data_cfg_value(cfg, 'sampling_rate_hz', None)
    if sampling_rate_hz is None:
        raise ValueError("sampling_rate_hz is required for MMFit dataset")
    sampling_rate_hz = int(sampling_rate_hz)
    
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
