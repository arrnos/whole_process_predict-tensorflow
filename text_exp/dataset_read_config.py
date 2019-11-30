#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python2.7
@author: zhangmeng
@file: dataset_read_config.py
@time: 2019/11/01
"""
from utils.make_bench_mark_data import chat_num_ls
from utils import tf_recorder_util
from config.global_config_text import *
import tensorflow as tf
import os

benchmark_file_base = "total_chat_num"
chat_num_ls = chat_num_ls
evaluate_batch_size = 1024

# dateset 加载配置
# tfRecorder = tf_recorder_util.TFRecorder()
# feature_ls = ["wechat_record"]
# label_name = "label"
# padding = ({"wechat_record": [SEQUENCE_MAX_LEN]}, [])

tfRecorder = tf_recorder_util.TFRecorder()
feature_ls = ["wechat_record","hist_student_chat_num", "hist_teacher_chat_num", "hist_total_chat_num"]
label_name = "label"
padding = ({"wechat_record": [SEQUENCE_MAX_LEN],"hist_student_chat_num":[], "hist_teacher_chat_num":[], "hist_total_chat_num":[],}, [])




if __name__ == '__main__':
    # 测试dataset
    test_tf_record_path = os.path.join(TF_RECORD_PATH, "wechat_record_feature")
    start_date = "20190803"
    end_date = "20190803"
    valid_dataset = tfRecorder.get_dataset_from_path(test_tf_record_path, feature_ls, label_name=label_name,
                                                     start_date=start_date, end_date=end_date,
                                                     batch_size=1, padding=padding)

    for i, d in enumerate(valid_dataset):
        print(d)
        if i > 300:
            break
