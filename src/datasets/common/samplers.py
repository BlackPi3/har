"""
Common sampling strategies for HAR datasets.
These samplers can be used across different datasets.
"""
import torch
from torch.utils.data import Sampler


class RandomStridedSampler(Sampler):
    """
    Randomly samples indices from a dataset with a given stride.
    Useful for training when you want randomization but also want to skip samples.
    """
    def __init__(self, dataset, stride_samples):
        """
        Args:
            dataset: Dataset to sample from
            stride_samples: Stride between samples (e.g., stride=10 means take every 10th sample)
        """
        self.dataset = dataset
        self.stride = stride_samples

    def __len__(self):
        return len(range(0, len(self.dataset), self.stride))

    def __iter__(self):
        return iter(torch.randperm(len(self.dataset))[: len(self)])


class SequentialStridedSampler(Sampler):
    """
    Sequentially samples indices from a dataset with a given stride.
    Useful for validation/testing when you want deterministic sampling.
    """
    def __init__(self, dataset, stride_samples):
        """
        Args:
            dataset: Dataset to sample from  
            stride_samples: Stride between samples (e.g., stride=10 means take every 10th sample)
        """
        self.dataset = dataset
        self.stride = stride_samples

    def __len__(self):
        return len(range(0, len(self.dataset), self.stride))

    def __iter__(self):
        return iter(range(0, len(self.dataset), self.stride))
