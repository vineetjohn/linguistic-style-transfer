import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

with open(global_config.label_mapped_style_embeddings_path, 'rb') as pickle_file:
    label_mapped_style_embeddings = pickle.load(pickle_file)
logger.info("Unpickled embeddings")

style_embeddings = list()
markers = list()
for label in label_mapped_style_embeddings:
    markers.append(len(style_embeddings))
    for style_embedding in label_mapped_style_embeddings[label]:
        style_embeddings.append(style_embedding)
markers.append(len(style_embeddings))
logger.debug("markers: {}".format(markers))

style_embeddings = np.asarray(a=style_embeddings)
logger.info("Extracted individual embeddings")

logger.info("Learning plot co-ordinates")
style_coordinates = TSNE(n_components=2).fit_transform(X=style_embeddings)
logger.debug("style_coordinates.shape: {}".format(style_coordinates.shape))

with open(global_config.label_names_path, 'rb') as pickle_file:
    label_names = pickle.load(pickle_file)

colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
for i in range(len(label_mapped_style_embeddings)):
    plt.scatter(x=style_coordinates[markers[i]:markers[i + 1], 0],
                y=style_coordinates[markers[i]:markers[i + 1], 1],
                marker='x', c=colors[i], label=label_names[i + 1])

plt.legend(loc='best')
plt.show()
