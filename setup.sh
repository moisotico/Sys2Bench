#!/bin/bash

set -e

unzip ./LLMs_Planning.zip
PLANNER_DIR="./LLMs-Planning/planner_tools"

# Check for VAL directory
VAL_DIR="$PLANNER_DIR/VAL"
if [ -d "$VAL_DIR" ]; then
    echo "VAL directory found. Setting up environment variable."
    export VAL="$VAL_DIR"
    echo "export VAL=$VAL_DIR" >> ~/.bashrc
    echo "export VAL=$VAL_DIR" >> ~/.bash_profile
else
    echo "Error: VAL directory not found!"
    exit 1
fi

# Check for PR2 directory
PR2_DIR="$PLANNER_DIR/PR2"
if [ -d "$PR2_DIR" ]; then
    echo "PR2 directory found. Setting up environment variable."
    export PR2="$PR2_DIR"
    echo "export PR2=$PR2_DIR" >> ~/.bashrc
    echo "export PR2=$PR2_DIR" >> ~/.bash_profile
else
    echo "Error: PR2 directory not found!"
    exit 1
fi

# Check and activate Conda environment
if [ -f "sys2bench.yaml" ]; then
    echo "Creating and activating Conda environment from sys2bench.yaml..."
    conda env create -f sys2bench.yaml --name sys2bench || echo "Environment already exists."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate sys2bench
    echo "Conda environment sys2bench activated."
else
    echo "Error: sys2bench.yaml not found!"
    exit 1
fi

echo "Setup completed successfully!"
echo "VAL is set to: $VAL"
echo "PR2 is set to: $PR2"
echo "Active Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
