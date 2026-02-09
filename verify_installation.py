"""
Quick script to verify all key packages are installed correctly.
Run this after setting up the virtual environment.
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"[OK] {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name}: NOT INSTALLED - {e}")
        return False

print("=" * 60)
print("Verifying Package Installation")
print("=" * 60)
print(f"Python version: {sys.version}\n")

packages = [
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('scipy', 'SciPy'),
    ('sklearn', 'scikit-learn'),
    ('torch', 'PyTorch'),
    ('torch_geometric', 'PyTorch Geometric'),
    ('torchvision', 'torchvision'),
    ('networkx', 'NetworkX'),
    ('xarray', 'xarray'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('plotly', 'plotly'),
    ('jupyter', 'Jupyter'),
    ('jupyterlab', 'JupyterLab'),
    ('requests', 'requests'),
    ('openmeteo_requests', 'openmeteo-requests'),
    ('optuna', 'Optuna'),
    ('tqdm', 'tqdm'),
    ('tensorboard', 'TensorBoard'),
]

results = []
for module, name in packages:
    results.append(check_import(module, name))

print("\n" + "=" * 60)
success_count = sum(results)
total_count = len(results)
print(f"Summary: {success_count}/{total_count} packages installed successfully")

if success_count == total_count:
    print("[SUCCESS] All packages installed correctly!")
    sys.exit(0)
else:
    print("[WARNING] Some packages are missing. Please check the errors above.")
    sys.exit(1)
