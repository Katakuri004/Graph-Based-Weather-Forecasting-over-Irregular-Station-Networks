"""
Utility functions for path resolution in notebooks.
"""

import sys
import os
from pathlib import Path


def get_project_root():
    """
    Get the project root directory.
    
    This function tries multiple methods to find the project root:
    1. Checks if we're in a known subdirectory and goes up
    2. Looks for project root markers (like .git, README.md, src/)
    3. Falls back to current working directory
    
    Returns:
    --------
    Path
        Path to project root
    """
    current_dir = Path(os.getcwd()).resolve()
    
    # Method 1: Check if we're in a known subdirectory
    if current_dir.name == '01_data_acquisition' or current_dir.name == '02_data_preprocessing':
        # We're in a notebook subdirectory
        return current_dir.parent.parent
    elif current_dir.name == 'notebooks':
        return current_dir.parent
    elif (current_dir / 'src').exists() and (current_dir / 'notebooks').exists():
        # We're already at project root
        return current_dir
    
    # Method 2: Walk up the directory tree looking for project markers
    for parent in current_dir.parents:
        if (parent / 'src').exists() and (parent / 'notebooks').exists():
            return parent
    
    # Method 3: Fall back to current directory
    return current_dir


def setup_project_path():
    """
    Add project root to Python path and return the project root.
    
    Returns:
    --------
    Path
        Path to project root
    """
    project_root = get_project_root()
    
    # Add to Python path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root


if __name__ == "__main__":
    # Test the function
    root = setup_project_path()
    print(f"Project root: {root}")
    print(f"src exists: {(root / 'src').exists()}")
    print(f"notebooks exists: {(root / 'notebooks').exists()}")
