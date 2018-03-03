import logging


def setup_custom_logger(name, log_level):

    formatter = logging.Formatter(fmt='%(asctime)s:%(levelname)s: %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.addHandler(handler)

    return logger
