import os
import re

import nltk

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

data_folder = "data/c50/training-set"
text_file_path = "data/c50/articles.txt"
labels_file_path = "data/c50/labels.txt"


def clean_text(string):
    string = re.sub(r"\\n", " ", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r'\d+', "number", string)
    string = string.replace("\r", " ")
    string = string.replace("\n", " ")
    string = string.strip().lower()

    return string


authors = os.listdir(data_folder)
logger.debug("Authors: {}".format(authors))

article_list = list()
author_labels = list()
file_extension = ".txt"

with open(text_file_path, 'w') as text_file, open(labels_file_path, 'w') as label_file:
    for author in authors:
        author_directory = data_folder + "/" + author
        files = os.listdir(author_directory)

        for filepath in map(lambda x: author_directory + "/" + x, files):
            if filepath[-1 * len(file_extension):] == file_extension:
                with open(filepath, 'r') as file:
                    article = file.read()
                    sentences = nltk.tokenize.sent_tokenize(article)
                    for sentence in sentences:
                        cleaned_sentence = clean_text(sentence)
                        text_file.write(cleaned_sentence + "\n")
                        label_file.write(author + "\n")
