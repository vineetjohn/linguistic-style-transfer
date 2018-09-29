import re
import os
import json
import random

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

config_folder = "data/shuttersong/config"
lyric_folder = "data/shuttersong/lyric"

train_text_file_path = "data/shuttersong/lyric-train.txt"
train_label_file_path = "data/shuttersong/song_artist-train.txt"
val_text_file_path = "data/shuttersong/lyric-val.txt"
val_label_file_path = "data/shuttersong/song_artist-val.txt"
test_text_file_path = "data/shuttersong/lyric-test.txt"
test_label_file_path = "data/shuttersong/song_artist-test.txt"
all_text_file_path = "data/shuttersong/lyric-all.txt"
all_label_file_path = "data/shuttersong/song_artist-all.txt"

song_artists_file_path = "data/shuttersong/song_artists.tsv"


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


def get_stanzas_from_song(song_text):
    song_text = re.sub(r'\n\s+\n', '\n\n', song_text)
    return song_text.split('\n\n')


max_line_length = global_config.max_sequence_length
all_lyric_tuples = list()
count = 0
song_artists = dict()
for config_file_name, lyric_file_name in zip(os.listdir(path=config_folder), os.listdir(path=lyric_folder)):
    with open(os.path.join(config_folder, config_file_name), 'r') as config_file, \
            open(os.path.join(lyric_folder, lyric_file_name), 'r') as lyric_file:
        config_json = json.load(config_file)
        song_artist = config_json['song_artist']

        if song_artist not in song_artists:
            song_artists[song_artist] = 0
        song_artists[song_artist] += 1
        count += 1
        # if count > 10:
        #     break


with open(song_artists_file_path, 'w') as song_artists_file:
    for agg_song_artist in sorted(song_artists.items(), key=lambda x: x[1], reverse=True):
        song_artists_file.write("{}\t{}\n".format(agg_song_artist[0], agg_song_artist[1]))
print(count)
exit()

total_size = len(all_lyric_tuples)
val_size = int(dev_proportion * total_size)
test_size = int(test_proportion * total_size)
logger.info("total_size: {}".format(total_size))
logger.info("val_size: {}".format(val_size))
logger.info("test_size: {}".format(test_size))
random.shuffle(all_lyric_tuples)

val_set = all_lyric_tuples[:val_size]
test_set = all_lyric_tuples[val_size:val_size + test_size]
train_set = all_lyric_tuples[val_size + test_size:]

with open(val_text_file_path, 'w') as text_file, \
        open(val_label_file_path, 'w') as artists_file:
    for lyric, artist in val_set:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))

with open(test_text_file_path, 'w') as text_file, \
        open(test_label_file_path, 'w') as artists_file:
    for lyric, artist in test_set:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))

with open(train_text_file_path, 'w') as text_file, \
        open(train_label_file_path, 'w') as artists_file:
    for lyric, artist in train_set:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))

with open(all_text_file_path, 'w') as text_file, \
        open(all_label_file_path, 'w') as artists_file:
    for lyric, artist in all_lyric_tuples:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))

logger.info("Processing complete")
