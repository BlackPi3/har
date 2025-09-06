"""
Common utilities for HAR dataset loading and processing.
These utilities can be shared across different datasets.
"""
import numpy as np
import csv
from pathlib import Path
from typing import Optional, List, Any


def load_numpy_file(filepath: str) -> Optional[np.ndarray]:
    """
    Safely loads a numpy file and returns the array, or None if file not found.
    
    Args:
        filepath: Path to the numpy file
        
    Returns:
        Numpy array if successful, None if file not found
    """
    try:
        return np.load(filepath)
    except FileNotFoundError as e:
        print(f"{e}. Returning None")
        return None


def load_csv_file(filepath: str) -> List[List[Any]]:
    """
    Loads a CSV file and returns a list of rows.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        List of lists containing the CSV data
    """
    data = []
    try:
        with open(filepath, "r") as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                data.append(line)
    except FileNotFoundError as e:
        print(f"Error loading CSV file {filepath}: {e}")
        raise
    return data


def validate_file_exists(filepath: str, file_type: str = "file") -> bool:
    """
    Validates that a file exists and provides informative error messages.
    
    Args:
        filepath: Path to validate
        file_type: Description of file type for error messages
        
    Returns:
        True if file exists, raises FileNotFoundError otherwise
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"{file_type} not found: {filepath}")
    return True


def ensure_directory_exists(directory: str) -> Path:
    """
    Ensures a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
