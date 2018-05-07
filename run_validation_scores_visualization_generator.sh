#!/usr/bin/env bash

source $HOME/.pyenv/bin/activate

PROJECT_DIR_PATH="$PWD/$(dirname $0)/"
cd ${PROJECT_DIR_PATH}

export PYTHONPATH=${PROJECT_DIR_PATH}

python -u linguistic_style_transfer_model/visualizers/scores_visualizer.py "$@"
