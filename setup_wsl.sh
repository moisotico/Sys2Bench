#!/bin/bash

# WSL-compatible setup script for Sys2Bench
# This script is designed to work in Windows Subsystem for Linux (WSL)

set -e

echo "Starting WSL setup for Sys2Bench..."

# Unzip LLMs_Planning.zip
if [ -f "./LLMs_Planning.zip" ]; then
    echo "Extracting LLMs_Planning.zip..."
    unzip -o ./LLMs_Planning.zip
else
    echo "Warning: LLMs_Planning.zip not found!"
fi

PLANNER_DIR="./LLMs-Planning/planner_tools"

# Check for VAL directory
VAL_DIR="$PLANNER_DIR/VAL"
if [ -d "$VAL_DIR" ]; then
    echo "VAL directory found. Setting up environment variable."
    VAL_ABSOLUTE=$(realpath "$VAL_DIR")
    export VAL="$VAL_ABSOLUTE"
    
    # Add to shell profile files if they exist, create if they don't
    for profile_file in ~/.bashrc ~/.bash_profile ~/.profile; do
        if [ -f "$profile_file" ] || [ "$profile_file" = ~/.bashrc ]; then
            # Remove any existing VAL exports to avoid duplicates
            sed -i '/export VAL=/d' "$profile_file" 2>/dev/null || true
            echo "export VAL=$VAL_ABSOLUTE" >> "$profile_file"
        fi
    done
    
    echo "VAL environment variable set to: $VAL_ABSOLUTE"
else
    echo "Error: VAL directory not found at $VAL_DIR!"
    exit 1
fi

# Check for PR2 directory
PR2_DIR="$PLANNER_DIR/PR2"
if [ -d "$PR2_DIR" ]; then
    echo "PR2 directory found. Setting up environment variable."
    PR2_ABSOLUTE=$(realpath "$PR2_DIR")
    export PR2="$PR2_ABSOLUTE"
    
    # Add to shell profile files if they exist, create if they don't
    for profile_file in ~/.bashrc ~/.bash_profile ~/.profile; do
        if [ -f "$profile_file" ] || [ "$profile_file" = ~/.bashrc ]; then
            # Remove any existing PR2 exports to avoid duplicates
            sed -i '/export PR2=/d' "$profile_file" 2>/dev/null || true
            echo "export PR2=$PR2_ABSOLUTE" >> "$profile_file"
        fi
    done
    
    echo "PR2 environment variable set to: $PR2_ABSOLUTE"
else
    echo "Error: PR2 directory not found at $PR2_DIR!"
    exit 1
fi

# Initialize conda in WSL if not already done
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH. Attempting to initialize..."
    
    # Common conda installation paths in WSL
    CONDA_PATHS=(
        "/home/$USER/miniconda3"
        "/home/$USER/anaconda3"
        "/opt/miniconda3"
        "/opt/anaconda3"
        "/usr/local/miniconda3"
        "/usr/local/anaconda3"
    )
    
    CONDA_FOUND=false
    for conda_path in "${CONDA_PATHS[@]}"; do
        if [ -d "$conda_path" ]; then
            echo "Found conda installation at: $conda_path"
            source "$conda_path/etc/profile.d/conda.sh"
            CONDA_FOUND=true
            break
        fi
    done
    
    if [ "$CONDA_FOUND" = false ]; then
        echo "Error: Could not find conda installation!"
        echo "Please install miniconda or anaconda in WSL, or add conda to your PATH."
        echo "You can install miniconda with:"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  bash Miniconda3-latest-Linux-x86_64.sh"
        exit 1
    fi
fi

# Check and create Conda environment
if [ -f "sys2bench.yaml" ]; then
    echo "Creating Conda environment from sys2bench.yaml..."
    
    # Check if environment already exists
    if conda env list | grep -q "^sys2bench "; then
        echo "Environment 'sys2bench' already exists. Updating..."
        conda env update -f sys2bench.yaml --name sys2bench
    else
        echo "Creating new environment 'sys2bench'..."
        conda env create -f sys2bench.yaml --name sys2bench
    fi
    
    echo "Conda environment 'sys2bench' is ready."
else
    echo "Error: sys2bench.yaml not found!"
    exit 1
fi

echo ""
echo "Setup completed successfully!"
echo "VAL is set to: $VAL"
echo "PR2 is set to: $PR2"
echo ""
echo "To activate the conda environment, run:"
echo "  conda activate sys2bench"
echo ""
echo "Note: You may need to restart your terminal or run 'source ~/.bashrc' for changes to take effect."
