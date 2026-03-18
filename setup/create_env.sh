#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=env_creation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=01:00:00

# Exit immediately if a command exits with a non-zero status
set -e

# Resolve repository root from script location.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

# Start timer
start_time=$(date +%s)

# Create environment name with the current date
env_prefix=stem

# Create the conda environment

source $CONDA_ROOT/etc/profile.d/conda.sh
conda create -n $env_prefix python=3.11 -y -c anaconda
conda activate $env_prefix

echo "Currently in env $(which python)"

# Install packages
pip install torch==2.7.0 xformers
pip install ninja
pip install --requirement "$repo_root/requirements.txt"

# Use a local lm-eval-harness checkout rather than the PyPI lm-eval wheel.
if [ ! -d "$repo_root/lm-evaluation-harness" ]; then
    git clone --depth 1 --branch v0.4.10 https://github.com/EleutherAI/lm-evaluation-harness.git "$repo_root/lm-evaluation-harness"
fi
pip install -e "$repo_root/lm-evaluation-harness"

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"


