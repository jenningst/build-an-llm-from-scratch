from pathlib import Path
from typing import Any, Dict

def get_data_dir() -> Path:
    """Return the path to the data directory."""
    return Path(__file__).parent / 'data'

def load_text(filename: str) -> str:
    """
    Load a text file from the data directory.
    
    Args:
        filename: Name of the file to load
        
    Returns:
        Loaded text
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    data_path = get_data_dir() / filename
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    return data_path.read_text()