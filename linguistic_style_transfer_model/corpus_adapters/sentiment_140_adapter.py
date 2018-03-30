import re
import string

from nltk.tokenize import TweetTokenizer

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

source_file_path = "data/sentiment140/train-data.csv"
text_file_path = "data/sentiment140/tweets.txt"
labels_file_path = "data/sentiment140/sentiment.txt"

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

tknzr = TweetTokenizer(strip_handles=True)


def clean_word(word):
    word = "" if (len(word) > 3 and word[:4] == "http") else word
    word = "" if (any([char not in string.printable for char in word])) else word
    return word


def clean_sentence(s):
    tokens = tknzr.tokenize(s)
    s = " ".join([clean_word(x) for x in tokens])

    s = re.sub(r"\\n", " ", s)
    s = re.sub(r"\'m", " am", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r'\d+', "number", s)
    s = s.replace("\r", " ")
    s = s.replace("\n", " ")
    s = s.strip().lower()

    return s


with open(file=text_file_path, mode='w') as text_file, \
        open(file=labels_file_path, mode='w') as label_file, \
        open(file=source_file_path, mode='r', encoding='ISO-8859-1') as source_file:
    for line in source_file:
        try:
            split_line = line.split(",")
            cleaned_sentence = clean_sentence(split_line[-1][1:-2])
            text_file.write(cleaned_sentence + "\n")

            if int(split_line[0][1:-1]):
                label_file.write("pos\n")
            else:
                label_file.write("neg\n")
        except Exception:
            logger.debug("Skipped: {}".format(line))
