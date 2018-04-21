import json
import re

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

files = {
    "home-and-kitchen": "data/amazon-reviews-multi-domain/reviews_home_and_kitchen.json",
    "electronics": "data/amazon-reviews-multi-domain/reviews_electronics.json"
}
reviews_per_label = 1024
text_file_path = "data/amazon-reviews-multi-domain/reviews.txt"
labels_file_path = "data/amazon-reviews-multi-domain/sentiment.txt"
category_file_path = "data/amazon-reviews-multi-domain/category.txt"
ratings_to_find = {
    5: "pos",
    1: "neg"
}

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")


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


def get_corpus_stats():
    for category in files:
        rating_counts = dict()
        for i in range(1, 6):
            rating_counts[i] = 0

        with open(files[category], 'r') as raw_review_file:
            for json_review_string in raw_review_file:
                json_review = json.loads(json_review_string)
                rating = int(json_review["overall"])
                review = json_review["reviewText"]
                if len(review.split()) < 20:
                    rating_counts[rating] += 1

        logger.info("category: {}; stats: {}".format(category, rating_counts))


def clean_corpus():
    for rating_sought in ratings_to_find:
        for category in files:
            with open(text_file_path, 'a') as text_file, open(labels_file_path, 'a') as label_file, \
                    open(category_file_path, 'a') as category_file:
                with open(files[category], 'r') as raw_review_file:
                    count = 0
                    for json_review_string in raw_review_file:
                        json_review = json.loads(json_review_string)
                        rating = int(json_review["overall"])
                        review = json_review["reviewText"]
                        if rating == rating_sought and len(review.split()) < 20:
                            cleaned_sentence = clean_text(review)
                            if cleaned_sentence:
                                text_file.write(cleaned_sentence + "\n")
                                label_file.write(ratings_to_find[rating] + "\n")
                                category_file.write(category + "\n")
                                count += 1
                        if count == reviews_per_label:
                            break

            logger.info("Generated file with rating {} and category {}".format(rating_sought, category))


# get_corpus_stats()
clean_corpus()
