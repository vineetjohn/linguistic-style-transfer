#!/usr/bin/env bash

export PYTHONPATH=$HOME/projects/authorship-style-transfer/
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

source $HOME/.pyenv/bin/activate

python -u authorship_style_transfer_network/main.py
