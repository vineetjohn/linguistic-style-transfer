import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

with open(global_config.label_mapped_style_embeddings_path, 'rb') as pickle_file:
    label_mapped_style_embeddings = pickle.load(pickle_file)
logger.info("Unpicked embeddings")

labels = list()
style_embeddings = list()
for label in label_mapped_style_embeddings:
    for style_embedding in label_mapped_style_embeddings[label]:
        labels.append(label)
        style_embeddings.append(style_embedding)
style_embeddings = np.asarray(a=style_embeddings)
logger.info("Extracted individual embeddings")

logger.info("Learning plot co-ordinates")
style_coordinates = TSNE(n_components=2).fit_transform(X=style_embeddings)
logger.debug("style_coordinates.shape: {}".format(style_coordinates.shape))

plt.scatter(style_coordinates[:, 0], style_coordinates[:, 1], c=labels)

plt.legend()
plt.show()
