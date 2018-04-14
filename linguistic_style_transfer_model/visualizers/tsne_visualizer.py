import pickle

from matplotlib import pyplot as plt

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']


def plot_coordinates(coordinates, plot_path, markers, label_names):
    for i in range(len(markers) - 1):
        plt.scatter(x=coordinates[markers[i]:markers[i + 1], 0],
                    y=coordinates[markers[i]:markers[i + 1], 1],
                    marker='x', c=colors[i], label=label_names[i + 1], alpha=0.5)

    plt.legend(loc='best')
    plt.savefig(fname=plot_path, format="svg", dpi=1200)


def main():
    with open(global_config.label_names_path, 'rb') as pickle_file:
        label_names = pickle.load(pickle_file)
    logger.info("label_names: {}".format(label_names))

    with open(global_config.style_coordinates_path, 'rb') as pickle_file:
        (style_coordinates, markers) = pickle.load(pickle_file)
        plot_coordinates(style_coordinates, global_config.style_embedding_plot_path,
                         markers, label_names)

    with open(global_config.content_coordinates_path, 'rb') as pickle_file:
        (content_coordinates, markers) = pickle.load(pickle_file)
        plot_coordinates(content_coordinates, global_config.content_embedding_plot_path,
                         markers, label_names)


if __name__ == "__main__":
    main()
