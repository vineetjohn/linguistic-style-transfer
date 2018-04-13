import pickle

from matplotlib import pyplot as plt

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

with open(global_config.label_names_path, 'rb') as pickle_file:
    label_names = pickle.load(pickle_file)
logger.info("label_names: {}".format(label_names))

with open(global_config.style_coordinates_path, 'rb') as pickle_file:
    (style_coordinates, markers) = pickle.load(pickle_file)

colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
for i in range(len(markers) - 1):
    plt.scatter(x=style_coordinates[markers[i]:markers[i + 1], 0],
                y=style_coordinates[markers[i]:markers[i + 1], 1],
                marker='x', c=colors[i], label=label_names[i + 1], alpha=0.5)

plt.legend(loc='best')
plt.savefig(fname=global_config.style_embedding_plot_path, format="svg", dpi=1200)
