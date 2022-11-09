import logging
import sys

from config import general_config


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(general_config.logging.level)
    formatter = logging.Formatter(general_config.logging.format)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
