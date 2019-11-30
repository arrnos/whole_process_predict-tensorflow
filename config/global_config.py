# -*- coding: utf-8 -*-

"""
@author: liutao
@file: global_config.py
@time: 2019/10/9 17:24
"""

import os
import getpass

# user = getpass.getuser()
user = "liutao"

PARENT_DATA_DIR = "/home/%s/project_data/whole_process_predict-tensorflow" % user
VOCABULARY_DIR = os.path.join(PARENT_DATA_DIR, "vocabulary_file")
MinMaxValue_DIR = os.path.join(PARENT_DATA_DIR, "MinMaxValue_file")
# VOCABULARY_DIR = "C:\\Users\\L\\PycharmProjects\\whole_process_predict-tensorflow\\data\\voacabulary_file"
