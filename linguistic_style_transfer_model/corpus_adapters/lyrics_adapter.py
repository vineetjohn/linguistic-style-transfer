import json
import random
import re

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

raw_lyrics_file_path = "data/lyrics/artist-song-line.top30artists.txt"
genre_mapping_file_path = "data/lyrics/artist-genres.json"

val_text_file_path = "data/lyrics/lyrics-val.txt"
val_artists_file_path = "data/lyrics/artist-val.txt"
val_genres_file_path = "data/lyrics/genre-val.txt"

test_text_file_path = "data/lyrics/lyrics-test.txt"
test_artists_file_path = "data/lyrics/artist-test.txt"
test_genres_file_path = "data/lyrics/genre-test.txt"

train_text_file_path = "data/lyrics/lyrics-train.txt"
train_artists_file_path = "data/lyrics/artist-train.txt"
train_genres_file_path = "data/lyrics/genre-train.txt"

all_text_file_path = "data/lyrics/lyrics-all.txt"
all_artists_file_path = "data/lyrics/artist-all.txt"
all_genres_file_path = "data/lyrics/genre-all.txt"

dev_proportion = 0.01
test_proportion = 0.05


def clean_text(string):
    string = re.sub(r"\d+", "", string)

    string = string.replace(".", "")
    string = string.replace("(", "")
    string = string.replace(")", "")
    string = string.replace("'m", " am")
    string = string.replace("'s", " is")
    string = string.replace("'ve", " have")
    string = string.replace("n't", " not")
    string = string.replace("'re", " are")
    string = string.replace("'d", " would")
    string = string.replace("'ll", " will")
    string = string.replace("\r", " ")
    string = string.replace("\n", " ")
    string = string.strip().lower()

    return string


def clean_lyric(lyric):
    lyric = clean_text(lyric)
    split_lyric = lyric.split()
    return lyric if len(split_lyric) > 5 else None


all_lyrics_tuples = list()
with open(raw_lyrics_file_path, 'r') as lyrics_file:
    for line in lyrics_file:
        (lyric, artist, _) = line.split(sep=",")
        lyric = clean_lyric(lyric)
        if lyric:
            all_lyrics_tuples.append((lyric.strip(), artist.strip()))

with open(genre_mapping_file_path) as genre_mapping_file:
    genre_map = json.load(genre_mapping_file)

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

with open(val_text_file_path, 'w') as text_file, \
        open(val_artists_file_path, 'w') as artists_file, \
        open(val_genres_file_path, 'w') as genres_file:
    for lyric, artist in val_set:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))
        genres_file.write("{}\n".format(genre_map[artist]))

with open(test_text_file_path, 'w') as text_file, \
        open(test_artists_file_path, 'w') as artists_file, \
        open(test_genres_file_path, 'w') as genres_file:
    for lyric, artist in test_set:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))
        genres_file.write("{}\n".format(genre_map[artist]))

with open(train_text_file_path, 'w') as text_file, \
        open(train_artists_file_path, 'w') as artists_file, \
        open(train_genres_file_path, 'w') as genres_file:
    for lyric, artist in train_set:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))
        genres_file.write("{}\n".format(genre_map[artist]))

with open(all_text_file_path, 'w') as text_file, \
        open(all_artists_file_path, 'w') as artists_file, \
        open(all_genres_file_path, 'w') as genres_file:
    for lyric, artist in all_lyrics_tuples:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))
        genres_file.write("{}\n".format(genre_map[artist]))

logger.info("Processing complete")
