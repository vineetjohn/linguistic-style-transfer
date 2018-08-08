import random
import re
import xml.etree.ElementTree as etree

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

input_xml_file = "data/yahoo-questions/FullOct2007.xml"

val_questions_file_path = "data/yahoo-questions/questions-val.txt"
val_topics_file_path = "data/yahoo-questions/topics-val.txt"

test_questions_file_path = "data/yahoo-questions/questions-test.txt"
test_topics_file_path = "data/yahoo-questions/topics-test.txt"

train_questions_file_path = "data/yahoo-questions/questions-train.txt"
train_topics_file_path = "data/yahoo-questions/topics-train.txt"

all_questions_file_path = "data/yahoo-questions/questions-all.txt"
all_topics_file_path = "data/yahoo-questions/topics-all.txt"

selected_topics = {'Science & Mathematics', 'Entertainment & Music', 'Politics & Government'}
max_len = 15

dev_proportion = 0.01
test_proportion = 0.05

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")


def clean_text(string):
    string = re.sub(r"\d+", "", string)

    string = string.replace(".", "")
    string = string.replace("(", "")
    string = string.replace(")", "")
    string = string.replace("\"", "")
    string = string.replace("?", "")
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


def is_valid(question, topic):
    return \
        topic in selected_topics and \
        len(question.split()) <= 15


def main():
    total_size = 0
    root = None
    question, topic = None, None
    with open(all_questions_file_path, 'w') as qf, open(all_topics_file_path, 'w') as tf:
        for event, elem in etree.iterparse(input_xml_file, events=('start', 'end')):
            logger.debug("event: {}, elemtag: {}".format(event, elem.tag))

            # keep track of the root element
            if event == 'start' and elem.tag == 'ystfeed':
                root = elem

            # track the data elements needed for the dataset
            if event == 'end' and elem.tag == 'subject':
                question = elem.text
            if event == 'end' and elem.tag == 'maincat':
                topic = elem.text.strip()
                # write data to file
                if is_valid(question, topic):
                    qf.write("{}\n".format(clean_text(question)))
                    tf.write("{}\n".format(topic))
                    total_size += 1

            # when a data instance is completely read, clear root
            if event == 'end' and elem.tag == 'vespaadd':
                root.clear()

    logger.info("{} questions read".format(total_size))

    val_size = int(dev_proportion * total_size)
    test_size = int(test_proportion * total_size)
    logger.info("total_size: {}".format(total_size))
    logger.info("val_size: {}".format(val_size))
    logger.info("test_size: {}".format(test_size))

    logger.info("Creating dataset splits ...")
    indices = list(range(total_size))
    random.shuffle(indices)

    val_indices = set(indices[:val_size])
    test_indices = set(indices[val_size:val_size + test_size])

    with open(all_questions_file_path) as qf, open(all_topics_file_path) as tf, \
            open(val_questions_file_path, 'w') as qfval, open(val_topics_file_path, 'w') as tfval, \
            open(test_questions_file_path, 'w') as qftest, open(test_topics_file_path, 'w') as tftest, \
            open(train_questions_file_path, 'w') as qftrain, open(train_topics_file_path, 'w') as tftrain:

        for index, (question, topic) in enumerate(zip(qf, tf)):
            if index in val_indices:
                qfile = qfval
                tfile = tfval
            elif index in test_indices:
                qfile = qftest
                tfile = tftest
            else:
                qfile = qftrain
                tfile = tftrain

            qfile.write("{}".format(question))
            tfile.write("{}".format(topic))

    logger.info("Finished data preprocessing")


if __name__ == '__main__':
    main()
