"""
Base dataset class for HAR datasets with common functionality.
"""
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseHARDataset(Dataset, ABC):
    """
    Base class for Human Activity Recognition datasets.
    Provides common functionality that can be shared across datasets.
    """
    
    def __init__(self):
        super().__init__()
        self.ACTIONS: Dict[str, int] = {}  # To be defined by subclasses
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Get a data sample by index."""
        pass
    
    def plot_joint(self, data: torch.Tensor, joint_idx: int, label: Any = None, 
                   title: str = None) -> None:
        """
        Plot joint trajectory over time.
        
        Args:
            data: Tensor of shape (3, joints, time) or (3, time)
            joint_idx: Index of joint to plot (ignored if data is 2D)
            label: Optional label for the plot title
            title: Optional custom title
        """
        plt.close()
        
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # Handle both (3, joints, time) and (3, time) formats
        if len(data.shape) == 3:
            joint_data = data[:, joint_idx, :]
        elif len(data.shape) == 2:
            joint_data = data
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.5)
        
        ax[0].plot(joint_data[0, :], label="x", color='red')
        ax[0].set_ylabel('X')
        ax[0].legend()
        ax[0].grid(True)
        
        ax[1].plot(joint_data[1, :], label="y", color='green')
        ax[1].set_ylabel('Y')
        ax[1].legend()
        ax[1].grid(True)
        
        ax[2].plot(joint_data[2, :], label="z", color='blue')
        ax[2].set_ylabel('Z')
        ax[2].set_xlabel('Time')
        ax[2].legend()
        ax[2].grid(True)
        
        if title:
            fig.suptitle(title)
        elif label is not None:
            fig.suptitle(f'Joint: {joint_idx}, Activity: {label}')
        else:
            fig.suptitle(f'Joint: {joint_idx}')
        
        plt.show()
    
    def get_action_names(self) -> Dict[int, str]:
        """
        Get reverse mapping from action indices to names.
        
        Returns:
            Dictionary mapping action indices to action names
        """
        return {v: k for k, v in self.ACTIONS.items()}
    
    def get_num_classes(self) -> int:
        """
        Get the number of activity classes.
        
        Returns:
            Number of classes
        """
        return len(self.ACTIONS)
    
    def print_dataset_info(self) -> None:
        """Print basic information about the dataset."""
        print(f"Dataset: {self.__class__.__name__}")
        print(f"Size: {len(self)}")
        print(f"Classes: {self.get_num_classes()}")
        print(f"Action mapping: {self.ACTIONS}")
