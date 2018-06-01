import re

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

dev_pos_reviews_file_path = "data/yelp/sentiment.dev.1"
dev_neg_reviews_file_path = "data/yelp/sentiment.dev.0"
test_pos_reviews_file_path = "data/yelp/sentiment.test.1"
test_neg_reviews_file_path = "data/yelp/sentiment.test.0"
train_pos_reviews_file_path = "data/yelp/sentiment.train.1"
train_neg_reviews_file_path = "data/yelp/sentiment.train.0"

train_text_file_path = "data/yelp/reviews-train.txt"
train_labels_file_path = "data/yelp/sentiment-train.txt"
val_text_file_path = "data/yelp/reviews-val.txt"
val_labels_file_path = "data/yelp/sentiment-val.txt"
test_text_file_path = "data/yelp/reviews-test.txt"
test_labels_file_path = "data/yelp/sentiment-test.txt"


def clean_text(string):
    string = re.sub(r"\.", "", string)
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


logger.info("Writing validation dataset")
with open(val_text_file_path, 'w') as text_file, open(val_labels_file_path, 'w') as labels_file:
    with open(dev_pos_reviews_file_path, 'r') as reviews_file:
        for line in reviews_file:
            text_file.write(clean_text(line) + "\n")
            labels_file.write("pos" + "\n")
    with open(dev_neg_reviews_file_path, 'r') as reviews_file:
        for line in reviews_file:
            text_file.write(clean_text(line) + "\n")
            labels_file.write("neg" + "\n")

logger.info("Writing test dataset")
with open(test_text_file_path, 'w') as text_file, open(test_labels_file_path, 'w') as labels_file:
    with open(test_pos_reviews_file_path, 'r') as reviews_file:
        for line in reviews_file:
            text_file.write(clean_text(line) + "\n")
            labels_file.write("pos" + "\n")
    with open(test_neg_reviews_file_path, 'r') as reviews_file:
        for line in reviews_file:
            text_file.write(clean_text(line) + "\n")
            labels_file.write("neg" + "\n")

logger.info("Writing train dataset")
with open(train_text_file_path, 'w') as text_file, open(train_labels_file_path, 'w') as labels_file:
    with open(train_pos_reviews_file_path, 'r') as reviews_file:
        for line in reviews_file:
            text_file.write(clean_text(line) + "\n")
            labels_file.write("pos" + "\n")
    with open(train_neg_reviews_file_path, 'r') as reviews_file:
        for line in reviews_file:
            text_file.write(clean_text(line) + "\n")
            labels_file.write("neg" + "\n")

logger.info("Processing complete")
