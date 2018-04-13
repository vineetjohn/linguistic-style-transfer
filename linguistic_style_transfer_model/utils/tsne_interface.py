import logging
import pickle

import numpy as np
from sklearn.manifold import TSNE

from linguistic_style_transfer_model.config import global_config

logger = logging.getLogger(global_config.logger_name)


def generate_style_plot_coordinates(label_mapped_style_embeddings):
    style_embeddings = list()
    markers = list()
    for label in label_mapped_style_embeddings:
        markers.append(len(style_embeddings))
        for style_embedding in label_mapped_style_embeddings[label]:
            style_embeddings.append(style_embedding)
    markers.append(len(style_embeddings))
    logger.debug("markers: {}".format(markers))

    style_embeddings = np.asarray(a=style_embeddings)
    logger.info("Extracted individual embeddings of shape {}".format(style_embeddings.shape))

    logger.info("Learning plot co-ordinates")
    style_coordinates = \
        TSNE(n_components=2).fit_transform(X=style_embeddings) \
        if style_embeddings.shape[1] != 2 \
        else style_embeddings
    logger.debug("style_coordinates.shape: {}".format(style_coordinates.shape))

    with open(global_config.style_coordinates_path, 'wb') as pickle_file:
        pickle.dump((style_coordinates, markers), pickle_file)

    logger.info("Dumped T-SNE co-ordinates")
