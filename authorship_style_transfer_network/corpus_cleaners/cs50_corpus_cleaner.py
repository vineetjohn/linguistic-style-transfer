import os
import re

import nltk

data_folder = "data/training-set"
text_file_path = "data/c50-articles.txt"
labels_file_path = "data/c50-labels.txt"


def clean_text(string):
    string = re.sub(r"\\n", " ", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r'\d+', "num_placeholder", string)
    string = string.replace("\r", " ")
    string = string.replace("\n", " ")
    string = string.strip().lower()

    return string


authors = os.listdir(data_folder)
print(authors)

article_list = list()
author_labels = list()

with open(text_file_path, 'w') as text_file, open(labels_file_path, 'w') as label_file:
    for author in authors:
        author_directory = data_folder + "/" + author
        files = os.listdir(author_directory)

        for filepath in map(lambda x: author_directory + "/" + x,files):
            if filepath[-4:] == ".txt":
                with open(filepath, 'r') as file:
                    article = file.read()
                    sentences = nltk.tokenize.sent_tokenize(article)
                    for sentence in sentences:
                        cleaned_sentence = clean_text(sentence)
                        text_file.write(cleaned_sentence + "\n")
                        label_file.write(author + "\n")
