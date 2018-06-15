#!/usr/bin/env bash

PROJECT_DIR_PATH="$PWD/$(dirname $0)/../"
cd ${PROJECT_DIR_PATH}

PYTHONPATH=${PROJECT_DIR_PATH} \
python -u linguistic_style_transfer_model/visualizers/tsne_visualizer.py "$@"
