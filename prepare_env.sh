#!/bin/bash

# This script prepares the environment for the project
# It receives two arguments:
# 1. The weights to download
#   - 'base' for the base model weights
#   - 'large' for the large model weights
#   - 'all' for all weights
#   - 'none' for no weights
# 2. The datasets to download
#   - 'classification' for the classification datasets
#   - 'segmentation' for the segmentation datasets
#   - 'all' for all datasets
#   - 'none' for no datasets

tdiv='========================================================================='
bdiv='-------------------------------------------------------------------------'

# Function for printing a divider
# Args:
#  - $1: text to print
function step() {
    echo $tdiv
    echo $1
    echo $bdiv
}

step '🐍 Creating virtual environment...'
python -m venv venv  # if not already created


step '🐉 Activating virtual environment...'
source venv/bin/activate

step '📦 Upgrading pip...'
pip install --upgrade pip

step '📋 Installing requirements...'
pip install -r requirements.txt

step '🔚 Exiting virtual environment...'
deactivate

step '📥 Downloading model weights...'
if [ $1 == 'base' ] || [ $1 == 'all' ]; then
    echo '  🏝️ MIRAGE-Base'
    mkdir -p __weights
    curl --proto '=https' --tlsv1.2 -sSf https://github.com/j-morano/MIRAGE/releases/download/weights/MIRAGE-Base.pth
    mv MIRAGE-Base.pth __weights/
fi
if [ $1 == 'large' ] || [ $1 == 'all' ]; then
    echo '  🏝️ MIRAGE-Large'
    mkdir -p __weights
    curl --proto '=https' --tlsv1.2 -sSf https://github.com/j-morano/MIRAGE/releases/download/weights/MIRAGE-Large.pth
    mv MIRAGE-Large.pth __weights/
fi

step '📥 Downloading datasets...'
if [ $2 == 'classification' ] || [ $2 == 'all' ]; then
    echo '  📊 Classification datasets'
    mkdir -p __datasets
    curl --proto '=https' --tlsv1.2 -sSf
    # TODO
elif [ $2 == 'segmentation' ] || [ $2 == 'all' ]; then
    echo '  🩻 Segmentation datasets'
    mkdir -p __datasets
    # TODO
fi

step '🎉 Environment prepared!'
