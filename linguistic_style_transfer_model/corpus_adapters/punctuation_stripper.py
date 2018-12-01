import re
import sys
import argparse

from types import SimpleNamespace

from linguistic_style_transfer_model.utils import log_initializer
from linguistic_style_transfer_model.config import global_config


class Options(SimpleNamespace):

    def __init__(self):
        self.source_file_path = None
        self.target_file_path = None



def clean_text(string):
    
    string = string.replace(".", "")
    string = string.replace(".", "")
    string = string.replace("\n", " ")
    string = string.replace(" 's", " is")
    string = string.replace("'m", " am")
    string = string.replace("'ve", " have")
    string = string.replace("n't", " not")
    string = string.replace("'re", " are")
    string = string.replace("'d", " would")
    string = string.replace("'ll", " will")
    string = string.replace("\r", " ")
    string = string.replace("\n", " ")
    string = re.sub(r'\d+', "number", string)
    string = ''.join(x for x in string if x.isalnum() or x == " ")
    string = re.sub(r'\s{2,}', " ", string)
    string = string.strip().lower()

    return string


def strip_punctuation(source_file_path, target_file_path):
    with open(source_file_path, 'r') as source_file, \
            open(target_file_path, 'w') as target_file:
        for line in source_file:
            cleaned_line = clean_text(line)
            target_file.write("{}\n".format(cleaned_line))


def main(argv):
    options = Options()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file-path", type=str, required=True)
    parser.add_argument("--target-file-path", type=str, required=True)
    parser.parse_known_args(args=argv, namespace=options)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

    logger.info("Starting to clean source file")
    strip_punctuation(options.source_file_path, options.target_file_path)
    logger.info("Concluded cleaning source file")


if __name__ == "__main__":
    main(sys.argv[1:])
