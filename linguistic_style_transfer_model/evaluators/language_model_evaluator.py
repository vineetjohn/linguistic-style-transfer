import pickle
import statistics


def score_generated_sentences(generated_text_file_path, language_model_path):
    log_probs = list()
    with open(language_model_path, 'rb') as language_model_file:
        language_model = pickle.load(language_model_file)
        with open(generated_text_file_path) as generated_text_file:
            for sentence in generated_text_file:
                log_probs.append(language_model.score_sent(tuple(sentence.split())))

    return statistics.mean(log_probs)
