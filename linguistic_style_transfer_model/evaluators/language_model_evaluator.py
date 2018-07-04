import sys

import argparse
import pickle
import statistics

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = None


def score_generated_sentences(generated_text_file_path, language_model_path):
    log_probs = list()
    with open(language_model_path, 'rb') as language_model_file:
        language_model = pickle.load(language_model_file)
        with open(generated_text_file_path) as generated_text_file:
            for sentence in generated_text_file:
                log_probs.append(language_model.score_sent(tuple(sentence.split())))

    return statistics.mean(log_probs)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-text-file-path", type=str, required=True)
    parser.add_argument("--language-model-path", type=str, required=True)
    args_namespace = parser.parse_args(argv)
    command_line_args = vars(args_namespace)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

    ll_score = \
        score_generated_sentences(command_line_args['generated_text_file_path'],
                                  command_line_args['language_model_path'])
    logger.info("ll_score: {}".format(ll_score))


if __name__ == '__main__':
    main(sys.argv[1:])
