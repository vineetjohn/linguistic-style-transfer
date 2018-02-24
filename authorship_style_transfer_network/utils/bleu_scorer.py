from nltk.translate.bleu_score import corpus_bleu

BLEU_SCORE_WEIGHTS = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (0.34, 0.33, 0.33, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}


def get_corpus_bleu_scores(actual_word_lists, generated_word_lists):
    bleu_scores = dict()
    for i in range(1, 5):
        bleu_scores[i] = corpus_bleu(
            list_of_references=actual_word_lists,
            hypotheses=generated_word_lists,
            weights=BLEU_SCORE_WEIGHTS[i])

    return bleu_scores
