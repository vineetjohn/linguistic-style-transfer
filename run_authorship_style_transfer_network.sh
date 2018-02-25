#!/usr/bin/env bash

source $HOME/.pyenv/bin/activate

PROJECT_DIR_PATH="$PWD/$(dirname $0)/"
echo $PROJECT_DIR_PATH
cd $PROJECT_DIR_PATH

export PYTHONPATH=$PROJECT_DIR_PATH
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

python -u authorship_style_transfer_network/main.py

