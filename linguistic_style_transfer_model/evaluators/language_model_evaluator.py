import sys

import argparse
import pickle
import statistics
from types import SimpleNamespace

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = None


class Options(SimpleNamespace):

    def __init__(self):
        self.generated_text_file_path = "INFO"
        self.language_model_path = None
        self.use_kenlm = None


def score_generated_sentences(generated_text_file_path, language_model_path, use_kenlm):
    log_probs = list()

    if use_kenlm:
        import kenlm
        model = kenlm.LanguageModel(language_model_path)
        with open(generated_text_file_path) as generated_text_file:
            for sentence in generated_text_file:
                log_probs.append(model.score(sentence))
    else:
        with open(language_model_path, 'rb') as language_model_file:
            language_model = pickle.load(language_model_file)
            with open(generated_text_file_path) as generated_text_file:
                for sentence in generated_text_file:
                    log_probs.append(language_model.score_sent(tuple(sentence.split())))

    return statistics.mean(log_probs)


def main(argv):
    options = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-text-file-path", type=str, required=True)
    parser.add_argument("--language-model-path", type=str, required=True)
    parser.add_argument("--use-kenlm", action="store_true", default=False)

    parser.parse_known_args(args=argv, namespace=options)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

    ll_score = score_generated_sentences(
        options.generated_text_file_path, options.language_model_path, options.use_kenlm)
    logger.info("ll_score: {}".format(ll_score))


if __name__ == '__main__':
    main(sys.argv[1:])
