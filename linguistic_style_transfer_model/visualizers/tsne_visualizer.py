import argparse
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']


def plot_coordinates(coordinates, plot_path, markers, label_names, fig_num):
    plt.figure(fig_num)
    for i in range(len(markers) - 1):
        plt.scatter(x=coordinates[markers[i]:markers[i + 1], 0],
                    y=coordinates[markers[i]:markers[i + 1], 1],
                    marker='x', c=colors[i], label=label_names[i], alpha=0.5)

    plt.legend(loc='best')
    plt.savefig(fname=plot_path, format="svg", dpi=1200)


def plot_coordinates_with_custom_label(coordinates, labels, plot_path, fig_num):
    plt.figure(fig_num)
    current_color_index = 0

    label_coordinates = dict()
    for i in range(len(labels)):
        if labels[i] not in label_coordinates:
            label_coordinates[labels[i]] = list()
        label_coordinates[labels[i]].extend([coordinates[i]])

    for label in label_coordinates:
        current_coordinates = np.asarray(label_coordinates[label])
        plt.scatter(x=current_coordinates[:, 0], y=current_coordinates[:, 1],
                    marker='x', c=colors[current_color_index],
                    label=label, alpha=0.5)
        current_color_index += 1

    plt.legend(loc='best')
    plt.savefig(fname=plot_path, format="svg", dpi=1200)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-file-path", type=str, required=False)

    args = vars(parser.parse_args(args=argv))

    logger.info(args)
    if not args["label_file_path"]:
        with open(global_config.index_to_label_dict_path, 'rb') as pickle_file:
            label_names = pickle.load(pickle_file)
        logger.info("label_names: {}".format(label_names))

        with open(global_config.style_coordinates_path, 'rb') as pickle_file:
            (style_coordinates, markers) = pickle.load(pickle_file)
            plot_coordinates(style_coordinates, global_config.style_embedding_plot_path,
                             markers, label_names, 0)

        with open(global_config.content_coordinates_path, 'rb') as pickle_file:
            (content_coordinates, markers) = pickle.load(pickle_file)
            plot_coordinates(content_coordinates, global_config.content_embedding_plot_path,
                             markers, label_names, 1)
    else:
        labels = list()
        with open(file=args["label_file_path"], mode='r') as label_file:
            for label in label_file:
                labels.append(label)

        with open(global_config.style_coordinates_path, 'rb') as pickle_file:
            (style_coordinates, markers) = pickle.load(pickle_file)
            plot_coordinates_with_custom_label(
                style_coordinates, labels, global_config.style_embedding_custom_plot_path, 0)

        with open(global_config.content_coordinates_path, 'rb') as pickle_file:
            (content_coordinates, markers) = pickle.load(pickle_file)
            plot_coordinates_with_custom_label(
                content_coordinates, labels, global_config.content_embedding_custom_plot_path, 1)


if __name__ == "__main__":
    main(sys.argv[1:])
