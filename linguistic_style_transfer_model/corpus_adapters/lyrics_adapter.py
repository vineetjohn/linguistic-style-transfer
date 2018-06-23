import random
import re

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

raw_lyrics_file_path = "data/lyrics/artist-song-line.top30artists.txt"

val_text_file_path = "data/lyrics/lyrics-val.txt"
val_labels_file_path = "data/lyrics/artist-val.txt"
test_text_file_path = "data/lyrics/lyrics-test.txt"
test_labels_file_path = "data/lyrics/artist-test.txt"
train_text_file_path = "data/lyrics/lyrics-train.txt"
train_labels_file_path = "data/lyrics/artist-train.txt"
all_text_file_path = "data/lyrics/lyrics-all.txt"
all_labels_file_path = "data/lyrics/artist-all.txt"
dev_proportion = 0.01
test_proportion = 0.05


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


all_lyrics_tuples = list()
with open(raw_lyrics_file_path, 'r') as lyrics_file:
    for line in lyrics_file:
        (lyric, artist, _) = line.split(sep=",")
        all_lyrics_tuples.append((lyric, artist))

total_size = len(all_lyrics_tuples)
val_size = int(dev_proportion * total_size)
test_size = int(test_proportion * total_size)
logger.info("total_size: {}".format(total_size))
logger.info("val_size: {}".format(val_size))
logger.info("test_size: {}".format(test_size))
random.shuffle(all_lyrics_tuples)

val_set = all_lyrics_tuples[:val_size]
test_set = all_lyrics_tuples[val_size:val_size + test_size]
train_set = all_lyrics_tuples[val_size + test_size:]

with open(val_text_file_path, 'w') as text_file, open(val_labels_file_path, 'w') as labels_file:
    for lyric, artist in val_set:
        text_file.write("{}\n".format(lyric.strip()))
        labels_file.write("{}\n".format(artist.strip()))

with open(test_text_file_path, 'w') as text_file, open(test_labels_file_path, 'w') as labels_file:
    for lyric, artist in test_set:
        text_file.write("{}\n".format(lyric.strip()))
        labels_file.write("{}\n".format(artist.strip()))

with open(train_text_file_path, 'w') as text_file, open(train_labels_file_path, 'w') as labels_file:
    for lyric, artist in train_set:
        text_file.write("{}\n".format(lyric.strip()))
        labels_file.write("{}\n".format(artist.strip()))

with open(all_text_file_path, 'w') as text_file, open(all_labels_file_path, 'w') as labels_file:
    for lyric, artist in all_lyrics_tuples:
        text_file.write("{}\n".format(lyric.strip()))
        labels_file.write("{}\n".format(artist.strip()))

logger.info("Processing complete")
