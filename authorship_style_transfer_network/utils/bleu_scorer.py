from nltk.translate.bleu_score import corpus_bleu

from authorship_style_transfer_network.utils import global_constants


def get_corpus_bleu_scores(actual_word_lists, generated_word_lists):
    bleu_scores = dict()

    for i in range(len(global_constants.bleu_score_weights)):
        bleu_scores[i + 1] = corpus_bleu(
            list_of_references=actual_word_lists,
            hypotheses=generated_word_lists,
            weights=global_constants.bleu_score_weights[i + 1])

    return bleu_scores
