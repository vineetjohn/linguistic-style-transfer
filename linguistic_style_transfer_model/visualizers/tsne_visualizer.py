import sys

import argparse
import json
import matplotlib
import os
import pickle

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = None
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
plot_markers = ['x', '+']


def plot_coordinates(coordinates, plot_path, markers, label_names, fig_num):
    matplotlib.use('svg')
    import matplotlib.pyplot as plt

    plt.figure(fig_num)
    for i in range(len(markers) - 1):
        plt.scatter(x=coordinates[markers[i]:markers[i + 1], 0],
                    y=coordinates[markers[i]:markers[i + 1], 1],
                    marker=plot_markers[i % len(plot_markers)],
                    c=colors[i % len(colors)],
                    label=label_names[i], alpha=0.75)

    plt.legend(loc='upper right', fontsize='x-large')
    plt.axis('off')
    plt.savefig(fname=plot_path, format="svg", bbox_inches='tight', transparent=True)
    plt.close()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-model-path", type=str)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

    args = vars(parser.parse_args(args=argv))
    logger.info(args)

    with open(os.path.join(args["saved_model_path"],
                           global_config.index_to_label_dict_file), 'r') as file:
        label_names = json.load(file)
    logger.info("label_names: {}".format(label_names))

    with open(os.path.join(args["saved_model_path"],
                           global_config.style_coordinates_file), 'rb') as pickle_file:
        (style_coordinates, markers) = pickle.load(pickle_file)
        plot_coordinates(style_coordinates,
                         os.path.join(args["saved_model_path"],
                                      global_config.style_embedding_plot_file),
                         markers, label_names, 0)

    with open(os.path.join(args["saved_model_path"],
                           global_config.content_coordinates_file), 'rb') as pickle_file:
        (content_coordinates, markers) = pickle.load(pickle_file)
        plot_coordinates(content_coordinates,
                         os.path.join(args["saved_model_path"],
                                      global_config.content_embedding_plot_file),
                         markers, label_names, 1)


if __name__ == "__main__":
    main(sys.argv[1:])
