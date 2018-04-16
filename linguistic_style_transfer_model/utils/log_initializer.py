import logging


def setup_custom_logger(name, log_level):
    formatter = logging.Formatter(
        fmt="%(asctime)s: %(message)s",
        datefmt="%m-%dT%H:%M:%S")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.addHandler(handler)

    return logger
