import re
from random import shuffle

positive_reviews_file_path = "data/amazon-reviews/pos.txt"
negative_reviews_file_path = "data/amazon-reviews/neg.txt"

train_text_file_path = "data/amazon-reviews/reviews-train.txt"
train_labels_file_path = "data/amazon-reviews/sentiment-train.txt"
val_text_file_path = "data/amazon-reviews/reviews-val.txt"
val_labels_file_path = "data/amazon-reviews/sentiment-val.txt"
test_text_file_path = "data/amazon-reviews/reviews-test.txt"
test_labels_file_path = "data/amazon-reviews/sentiment-test.txt"

train_size = 65536
val_size = 1024
test_size = 16384


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


count = 0
total_reviews = train_size + val_size + test_size
print("Total Review size: {}".format(total_reviews))
collected_positive_reviews = list()
collected_negative_reviews = list()

with open(positive_reviews_file_path, 'r') as positive_reviews_file, \
        open(negative_reviews_file_path, 'r') as negative_reviews_file:
    for positive_review, negative_review in zip(positive_reviews_file, negative_reviews_file):
        collected_positive_reviews.append(clean_text(positive_review))
        collected_negative_reviews.append(clean_text(negative_review))

shuffle(collected_positive_reviews)
shuffle(collected_negative_reviews)


def write_file(text_file_path, labels_file_path, positive_reviews, negative_reviews):
    print("Positive reviews: {}, Negative reviews : {}".format(len(positive_reviews), len(negative_reviews)))
    with open(text_file_path, 'w') as text_file, open(labels_file_path, 'w') as label_file:
        for review in positive_reviews:
            text_file.write(review + "\n")
            label_file.write("pos" + "\n")
        for review in negative_reviews:
            text_file.write(review + "\n")
            label_file.write("neg" + "\n")


write_file(train_text_file_path, train_labels_file_path, collected_positive_reviews[:train_size],
           collected_negative_reviews[:train_size])
print("Training files saved")

write_file(val_text_file_path, val_labels_file_path, collected_positive_reviews[train_size:train_size + val_size],
           collected_negative_reviews[train_size:train_size + val_size])
print("Validation files saved")

write_file(test_text_file_path, test_labels_file_path,
           collected_positive_reviews[train_size + val_size:total_reviews],
           collected_negative_reviews[train_size + val_size:total_reviews])
print("Testing files saved")
