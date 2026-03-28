import launch
import importlib
import os
import subprocess
from packaging.version import Version
from packaging.requirements import Requirement

# Initialize sd_mecha submodule if not already done
_ext_dir = os.path.dirname(os.path.abspath(__file__))
_sd_mecha_dir = os.path.join(_ext_dir, "scripts", "sd_mecha")
if not os.path.isfile(os.path.join(_sd_mecha_dir, "setup.py")) and \
   not os.path.isfile(os.path.join(_sd_mecha_dir, "pyproject.toml")):
    print("sd-webui-supermerger: initializing sd_mecha submodule...")
    try:
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=_ext_dir,
            check=True,
            capture_output=True,
        )
        print("sd-webui-supermerger: sd_mecha submodule initialized.")
    except subprocess.CalledProcessError as e:
        print(f"sd-webui-supermerger: failed to initialize sd_mecha submodule: {e.stderr.decode().strip()}")
        print("  Run `git submodule update --init --recursive` in the extension directory manually.")

def is_installed(pip_package):
    """
    Check if a package is installed and meets version requirements specified in pip-style format.

    Args:
        pip_package (str): Package name in pip-style format (e.g., "numpy>=1.22.0").
    
    Returns:
        bool: True if the package is installed and meets the version requirement, False otherwise.
    """
    try:
        # Parse the pip-style package name and version constraints
        requirement = Requirement(pip_package)
        package_name = requirement.name
        specifier = requirement.specifier  # e.g., >=1.22.0
        
        # Check if the package is installed
        dist = importlib.metadata.distribution(package_name)
        installed_version = Version(dist.version)
        
        # Check version constraints
        if specifier.contains(installed_version):
            return True
        else:
            print(f"Installed version of {package_name} ({installed_version}) does not satisfy the requirement ({specifier}).")
            return False
    except importlib.metadata.PackageNotFoundError:
        print(f"Package {pip_package} is not installed.")
        return False
    
requirements = [
"diffusers==0.31.0",
"scikit-learn",
"accelerate",
"fuzzywuzzy",
"python-Levenshtein",
"scipy",
"PyYAML",
]

for module in requirements:
    if not is_installed(module):
        launch.run_pip(f"install {module}", module)