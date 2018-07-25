import csv
import random
import re

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

raw_lyrics_file_path = "data/lyrics_raw/songdata.csv"

val_text_file_path = "data/lyrics_raw/lyrics-val.txt"
val_artists_file_path = "data/lyrics_raw/artist-val.txt"
val_genres_file_path = "data/lyrics_raw/genre-val.txt"

test_text_file_path = "data/lyrics_raw/lyrics-test.txt"
test_artists_file_path = "data/lyrics_raw/artist-test.txt"
test_genres_file_path = "data/lyrics_raw/genre-test.txt"

train_text_file_path = "data/lyrics_raw/lyrics-train.txt"
train_artists_file_path = "data/lyrics_raw/artist-train.txt"
train_genres_file_path = "data/lyrics_raw/genre-train.txt"

all_text_file_path = "data/lyrics_raw/lyrics-all.txt"
all_artists_file_path = "data/lyrics_raw/artist-all.txt"
all_genres_file_path = "data/lyrics_raw/genre-all.txt"

dev_proportion = 0.01
test_proportion = 0.05

whitelisted_artists = {
    "Ella Fitzgerald",
    "Dolly Parton",
    "Rihanna",
    "Tori Amos"
}


# whitelisted_artists = {
#     "Pearl Jam",
#     "LL Cool J",
#     "Fleetwood Mac",
#     "Grateful Dead"
# }


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
all_lyrics_tuples = list()
with open(raw_lyrics_file_path, 'r') as lyrics_file:
    next(lyrics_file)
    csv_lyrics_reader = csv.reader(lyrics_file, delimiter=',', quotechar='"')
    for data_instance in csv_lyrics_reader:
        artist, title, link, song_text = data_instance
        if artist not in whitelisted_artists:
            continue
        stanzas = get_stanzas_from_song(song_text)
        # print("stanzas", stanzas)
        for stanza in stanzas:
            lines = set([x.strip() for x in stanza.split('\n')])
            # print("lines", lines)
            current_lines = list()
            cumulative_line_length = 0
            for line in lines:
                line = clean_text(line)
                words = line.split()
                line_length = len(words)
                if line_length < 3 or line_length > max_line_length:
                    continue
                cumulative_line_length += line_length
                if cumulative_line_length > max_line_length:
                    all_lyrics_tuples.append((" ".join(current_lines), artist))
                    cumulative_line_length = line_length
                    current_lines = list()
                current_lines.append(line)
            if cumulative_line_length > 0:
                all_lyrics_tuples.append((" ".join(current_lines), artist))

# print(all_lyrics_tuples[:10])
# exit()

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
        open(val_artists_file_path, 'w') as artists_file:
    for lyric, artist in val_set:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))

with open(test_text_file_path, 'w') as text_file, \
        open(test_artists_file_path, 'w') as artists_file:
    for lyric, artist in test_set:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))

with open(train_text_file_path, 'w') as text_file, \
        open(train_artists_file_path, 'w') as artists_file:
    for lyric, artist in train_set:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))

with open(all_text_file_path, 'w') as text_file, \
        open(all_artists_file_path, 'w') as artists_file:
    for lyric, artist in all_lyrics_tuples:
        text_file.write("{}\n".format(lyric))
        artists_file.write("{}\n".format(artist))

logger.info("Processing complete")
