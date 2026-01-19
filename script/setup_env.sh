#!/bin/bash
# Setup script for Neural-Classification-Trees on Compute Canada Rorqual cluster
# Run this once to create the virtual environment

set -e  # Exit on error

echo "=========================================="
echo "Setting up NCT Environment for Rorqual Cluster"
echo "=========================================="

# Define paths
PROJECT_DIR="/home/agarg/links/scratch/agarg/Neural-Classification-Trees/script"
ENV_DIR="$HOME/envs/nct_env"

# Load Python module (Python 3.11 is recommended for torch 2.9.1 compatibility)
module purge
module load python/3.11.5

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Create virtual environment directory if it doesn't exist
mkdir -p "$HOME/envs"

# Create virtual environment
if [ -d "$ENV_DIR" ]; then
    echo "Virtual environment already exists at $ENV_DIR"
    echo "To recreate, delete it first: rm -rf $ENV_DIR"
else
    echo "Creating virtual environment at $ENV_DIR..."
    virtualenv --no-download "$ENV_DIR"
fi

# Activate environment
source "$ENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip and packaging tools..."
pip install --no-index --upgrade --force-reinstall pip setuptools wheel

# Install other dependencies from Compute Canada wheels
echo "Installing other dependencies..."
pip install --no-index numpy pandas scikit-learn pillow tqdm matplotlib seaborn

# Install PyTorch and torchvision (use Compute Canada wheels)
echo "Installing PyTorch and torchvision..."
pip install torch torchvision

# Install captum and wandb
echo "Installing captum and wandb..."
pip install --no-index captum wandb

# Install any remaining packages that might not be in Compute Canada wheels
# These will be downloaded from PyPI if not available locally
echo "Installing remaining packages (may download from PyPI if not available)..."
pip install pyyaml

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
python -c "import captum; print(f'Captum version: {captum.__version__}')"
python -c "import wandb; print(f'Wandb version: {wandb.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in future sessions:"
echo "  module load python/3.11.5"
echo "  source $ENV_DIR/bin/activate"
echo ""
echo "Environment location: $ENV_DIR"
echo "=========================================="
