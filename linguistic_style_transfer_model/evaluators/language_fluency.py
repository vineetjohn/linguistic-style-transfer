import sys

import argparse
import statistics
from types import SimpleNamespace

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer
from linguistic_style_transfer_model.corpus_adapters.punctuation_stripper import clean_text

logger = None


class Options(SimpleNamespace):

    def __init__(self):
        self.generated_text_file_path = "INFO"
        self.language_model_path = None
        self.use_kenlm = None


def score_generated_sentences(generated_text_file_path, language_model_path):
    log_probs = list()
    perplexity_scores = list()

    import kenlm
    model = kenlm.LanguageModel(language_model_path)
    with open(generated_text_file_path) as generated_text_file:
        for sentence in generated_text_file:
            cleaned_sentence = clean_text(sentence)
            log_probs.append(model.score(cleaned_sentence))
            perplexity_scores.append(model.perplexity(cleaned_sentence))

    return statistics.mean(log_probs), statistics.mean(perplexity_scores)


def main(argv):
    options = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-text-file-path", type=str, required=True)
    parser.add_argument("--language-model-path", type=str, required=True)
    parser.add_argument("--use-kenlm", action="store_true", default=False)

    parser.parse_known_args(args=argv, namespace=options)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

    ll_score, perplexity_score = score_generated_sentences(
        options.generated_text_file_path, options.language_model_path)
    logger.info("ll_score: {}".format(ll_score))
    logger.info("perplexity_score: {}".format(perplexity_score))


if __name__ == '__main__':
    main(sys.argv[1:])
