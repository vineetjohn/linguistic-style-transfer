from nltk.translate.bleu_score import corpus_bleu

from linguistic_style_transfer_model.config import global_config


def get_corpus_bleu_scores(actual_word_lists, generated_word_lists):
    bleu_scores = dict()

    for i in range(len(global_config.bleu_score_weights)):
        bleu_scores[i + 1] = round(
            corpus_bleu(
                list_of_references=actual_word_lists[:len(generated_word_lists)],
                hypotheses=generated_word_lists,
                weights=global_config.bleu_score_weights[i + 1]), 4)

    return bleu_scores
