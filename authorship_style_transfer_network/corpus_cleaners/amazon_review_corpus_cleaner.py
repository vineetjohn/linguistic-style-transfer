import re

positive_reviews_file_path = "data/amazon-reviews/pos.txt"
negative_reviews_file_path = "data/amazon-reviews/neg.txt"
text_file_path = "data/amazon-reviews/reviews.txt"
labels_file_path = "data/amazon-reviews/sentiment.txt"


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


article_list = list()
author_labels = list()
file_extension = ".txt"

with open(text_file_path, 'w') as text_file, open(labels_file_path, 'w') as label_file:
    with open(positive_reviews_file_path, 'r') as positive_reviews_file:
        for review in positive_reviews_file:
            cleaned_sentence = clean_text(review)
            text_file.write(cleaned_sentence + "\n")
            label_file.write("pos" + "\n")

    with open(negative_reviews_file_path, 'r') as negative_reviews_file:
        for review in negative_reviews_file:
            cleaned_sentence = clean_text(review)
            text_file.write(cleaned_sentence + "\n")
            label_file.write("neg" + "\n")
