# Notebooks Directory

## Path Setup

All notebooks need to set up the project path correctly to import from `src`. 

### Quick Setup (Copy-Paste)

Add this to the first code cell of any notebook:

```python
import sys
import os
from pathlib import Path

# Find project root
current_dir = Path(os.getcwd()).resolve()
if (current_dir / 'src').exists() and (current_dir / 'notebooks').exists():
    project_root = current_dir
elif current_dir.name in ['01_data_acquisition', '02_data_preprocessing', '03_baselines', 
                           '04_gnn_models', '05_training', '06_analysis', '07_evaluation', '08_documentation']:
    project_root = current_dir.parent.parent
elif current_dir.name == 'notebooks':
    project_root = current_dir.parent
else:
    for parent in current_dir.parents:
        if (parent / 'src').exists() and (parent / 'notebooks').exists():
            project_root = parent
            break
    else:
        project_root = current_dir

# Add to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now you can import from src
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
```

### Alternative: Using the Utility Function

Once the path is set up, you can use:

```python
from src.utils.path_utils import setup_project_path
project_root = setup_project_path()
```

## Running Notebooks

1. **Start Jupyter Lab from project root:**
   ```bash
   cd C:\Users\Kata\Desktop\earth-sgnn
   jupyter lab
   ```

2. **Or ensure working directory is project root:**
   - In Jupyter, the working directory should be the project root
   - You can check with: `import os; print(os.getcwd())`

## Troubleshooting

### ModuleNotFoundError: No module named 'src'

1. Make sure you're running Jupyter from the project root directory
2. Check that `src/__init__.py` exists
3. Verify the path setup code is in your notebook
4. Restart the kernel and try again

### Import errors

- Ensure the virtual environment is activated
- Verify all packages are installed: `python verify_installation.py`
- Check that `src/utils/config.py` exists
