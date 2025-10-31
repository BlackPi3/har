"""
MMFit Dataset implementation.
"""
import bisect
import torch
from typing import Tuple
from ..common.base_dataset import BaseHARDataset
from .loaders import load_modality, load_labels
from .constants import (
    ACTIONS, JOINT_INDICES, POSE_DATA_INDICES, 
    DEFAULT_LABELING_TOLERANCE
)


class MMFit(BaseHARDataset):
    """
    MMFit dataset for pose-to-accelerometer regression and activity classification.
    
    Currently uses left wrist joint trajectory and accelerometer sensor data.
    """
    
    def __init__(self, pose_file: str, acc_file: str, labels_file: str,
                 sensor_window_length: int,
                 stride_seconds: float | None = None,
                 sampling_rate_hz: int = 100,
                 cluster: bool = False,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize MMFit dataset.
        
        Args:
            pose_file: Path to pose data file
            acc_file: Path to accelerometer data file  
            labels_file: Path to labels CSV file
            sensor_window_length: Length of the sliding window in samples
            cluster: Whether running on cluster (for data subset selection)
            dtype: PyTorch data type for tensors
        """
        super().__init__()
        
        self.sensor_window_length = int(sensor_window_length)
        # Stride: convert seconds to samples (assumes 100 Hz by default)
        if stride_seconds is None:
            self.stride = 1
        else:
            s = int(round(float(stride_seconds) * float(sampling_rate_hz)))
            self.stride = max(1, s)
        self.ACTIONS = ACTIONS
        
        # Load pose data: (3, N, joints) -> select specific joints
        pose_data = load_modality(pose_file)
        if pose_data is None:
            raise ValueError(f"Could not load pose data from {pose_file}")
        
        self.pose = torch.as_tensor(
            pose_data[:, :, JOINT_INDICES], dtype=dtype
        )  # (3, N, (frame, timestamps, left shoulder, left elbow, left wrist))
        
        # Load accelerometer data: (N, features)
        acc_data = load_modality(acc_file)
        if acc_data is None:
            raise ValueError(f"Could not load accelerometer data from {acc_file}")
            
        self.acc = torch.as_tensor(acc_data, dtype=dtype)  # (N, (frame, timestamps, XYZ))
        
        # For non-cluster runs, use subset for faster debugging
        if not cluster:
            select = 1000
            self.pose = self.pose[:, :select, :]
            self.acc = self.acc[:select, :]
        
        # Load and process labels
        self.labels = load_labels(labels_file)
        self.start_frames = [row[0] for row in self.labels]
        self.end_frames = [row[1] for row in self.labels]
    
    def __len__(self) -> int:
        """Return number of windows with hop 'stride'."""
        N = int(self.pose.shape[1])
        W = int(self.sensor_window_length)
        S = int(self.stride)
        if N <= W:
            return 0
        return (N - W) // S
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a data sample using dynamic windowing with hop size 'stride'.
        
        Args:
            idx: Sample index (0-based)
            
        Returns:
            Tuple of (pose_window, acc_window, activity_label)
                - pose_window: (3, joints, window_length) - pose trajectory 
                - acc_window: (3, window_length) - accelerometer data
                - activity_label: integer class label
        """
        # Compute window start/end using hop size
        S = int(self.stride)
        W = int(self.sensor_window_length)
        start = int(idx) * S
        end = start + W
        # Extract windowed data
        # Pose: (3, joints, W) - skip frame/timestamp columns and permute
        sample_pose = self.pose[:, start:end, POSE_DATA_INDICES].permute(0, 2, 1)
        # Accelerometer: (3, W) - skip frame/timestamp columns and permute
        sample_acc = self.acc[start:end, POSE_DATA_INDICES].permute(1, 0)
        
        # Determine activity label for this window
        window_start_frame = self.pose[0, start, 0]
        window_end_frame = self.pose[0, end, 0]
        
        # Use midpoint with tolerance for activity labeling
        tolerance = DEFAULT_LABELING_TOLERANCE
        mid_point = int(((window_end_frame - window_start_frame) * tolerance) + window_start_frame)
        
        # Find which activity (if any) contains this midpoint
        index = bisect.bisect_right(self.start_frames, mid_point)
        label = "non_activity"  # default
        
        if 0 < index and mid_point <= self.end_frames[index - 1]:
            label = self.labels[index - 1][3]
        
        return sample_pose, sample_acc, self.ACTIONS[label]
