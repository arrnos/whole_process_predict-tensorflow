#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: global_config.py
@time: 2019/10/22
"""

import os
import getpass
import platform

plat = ""
if platform.system().lower() == 'windows':
    plat = "windows"
elif platform.system().lower() == 'linux':
    plat = "linux"

assert plat in ("linux", "windows"), "platform is not linux or windows!"

PROJECT_NAME = "whole_process_predict-tensorflow"

if plat == "windows":
    os.chdir("..")
    PROJECT_DIR = os.getcwd()
    PROJECT_DATA_DIR = os.path.join("E:\project_data", PROJECT_NAME)
    MERGED_RAW_WECHAT_RECORD_PATH = "E:\project_data\%s\\raw_data\\merged_wechat_record_data" % PROJECT_NAME
    MERGED_RAW_DATA_PATH = "E:\project_data\%s\\raw_data\\merged_raw_data" % PROJECT_NAME

else:
    user = getpass.getuser()
    PROJECT_DIR = "/home/%s/%s" % (user, PROJECT_NAME)
    PROJECT_DATA_DIR = os.path.join("/home/%s/" % user, "project_data", PROJECT_NAME)
    MERGED_RAW_WECHAT_RECORD_PATH = "/home/yanxin/project_data/public_data/raw_data/merged_wechat_record_data"
    MERGED_RAW_DATA_PATH = "/home/yanxin/project_data/whole_process_extend_sample/merged_raw_data"

assert os.path.isdir(PROJECT_DATA_DIR), "%s不存在！" % PROJECT_DATA_DIR
assert os.path.isdir(PROJECT_DIR), "%s不存在！" % PROJECT_DIR

SEQUENCE_MAX_LEN = 500

VOCAB_PATH = os.path.join(PROJECT_DATA_DIR, "vocab.json")

TF_RECORD_PATH = os.path.join(PROJECT_DATA_DIR, "tf_record")
RAW_DATA_PATH = os.path.join(PROJECT_DATA_DIR, "raw_data")
FEATURE_PATH = os.path.join(PROJECT_DATA_DIR, "feature_file")
# 是否过滤掉微信文本为空的样本
FILTER_NULL_WECHAT_SAMPLE = True
