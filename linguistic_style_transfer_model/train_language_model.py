import sys

import argparse
import pickle
from nltk.util import ngrams
from types import SimpleNamespace

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.models.kn_language_model import KneserNeyLM
from linguistic_style_transfer_model.utils import log_initializer

logger = None


class Options(SimpleNamespace):

    def __init__(self):
        self.logging_level = "INFO"
        self.text_file_path = None
        self.model_save_path = None


def train_language_model(text_file_path, model_save_path):
    with open(text_file_path, 'r') as text_file:
        text_ngrams = list(
            ngram for sent in text_file
            for ngram in ngrams(sent.split(), global_config.language_model_order,
                                pad_left=True, pad_right=True,
                                left_pad_symbol='<s>', right_pad_symbol='</s>'))

    lm = KneserNeyLM(
        global_config.language_model_order, text_ngrams,
        start_pad_symbol='<s>', end_pad_symbol='</s>')

    with open(model_save_path, 'wb') as model_file:
        pickle.dump(lm, model_file)


def main(argv):
    options = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file-path", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--logging-level", type=str)

    parser.parse_known_args(args=argv, namespace=options)
    print(options)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, options.logging_level)

    train_language_model(options.text_file_path, options.model_save_path)

    logger.info("Language Model Training Complete!")


if __name__ == "__main__":
    main(sys.argv[1:])
