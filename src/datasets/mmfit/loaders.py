"""
Data loading utilities specific to the MMFit dataset.
"""
import csv
from typing import List, Optional
import numpy as np
from ..common.utils import load_numpy_file


def load_modality(filepath: str) -> Optional[np.ndarray]:
    """
    Loads MMFit modality from filepath and returns numpy array, or None if no file is found.
    
    Args:
        filepath: File path to MM-Fit modality file
        
    Returns:
        MM-Fit modality as numpy array, or None if file not found
    """
    return load_numpy_file(filepath)


def load_labels(filepath: str) -> List[List]:
    """
    Loads and reads MMFit CSV label file.
    
    Args:
        filepath: File path to a MM-Fit CSV label file
        
    Returns:
        List of lists containing label data: 
        (Start Frame, End Frame, Repetition Count, Activity) for each exercise set
    """
    labels = []
    try:
        with open(filepath, "r") as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                labels.append([int(line[0]), int(line[1]), int(line[2]), line[3]])
    except FileNotFoundError as e:
        print(f"Error loading labels file {filepath}: {e}")
        raise
    except (ValueError, IndexError) as e:
        print(f"Error parsing labels file {filepath}: {e}")
        raise
    
    return labels
