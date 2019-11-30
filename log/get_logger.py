# -*- coding: utf-8 -*-

"""
@author: liutao
@file: get_logger.py
@time: 2019/10/10 14:29
"""

import logging


def get_logger(logger_name, log_file):

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fomatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s: %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fomatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fomatter)
    console_handler.setLevel(logging.ERROR)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


LOG = get_logger("log", "log.txt")


if __name__ == '__main__':
    LOG.info("info")
    LOG.error("error")