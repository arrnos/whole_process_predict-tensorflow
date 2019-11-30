#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: transform_wechat_record_feature_2_TFRecord.py
@time: 2019/10/23
"""

from utils import tf_recorder_util
from config.global_config_text import *
from utils.date_util import DateUtil
from utils.text_transform_util import DataHelper

# feature transfer config
column_names = ["label", "opp_id", "create_time", "hist_student_chat_num",
                "hist_teacher_chat_num", "hist_total_chat_num", "wechat_record"]

need_features_cols = ["opp_id", "hist_student_chat_num", "hist_teacher_chat_num", "hist_total_chat_num",
                      "wechat_record"]

var_length_cols = ["wechat_record"]

col_preprocess_func = {
    "wechat_record": lambda text: dataHelper.transform_single_text_2_vector(text, SEQUENCE_MAX_LEN)
}

label_name = "label"
dataHelper = DataHelper()
tfRecorder = tf_recorder_util.TFRecorder()


def transfer_wechat_feature_2_tfRecord(start_date, end_date, raw_feature_file_path, tf_record_file_path):
    raw_feature_folder_name = os.path.basename(raw_feature_file_path)
    tf_record_folder_name = os.path.basename(tf_record_file_path)

    raw_feature_file = os.path.join(raw_feature_file_path, raw_feature_folder_name + "_%s")
    tf_record_file = os.path.join(tf_record_file_path, tf_record_folder_name + "_%s.tfrecord")
    data_info_csv_path = os.path.join(tf_record_file_path, "data_info.csv")

    if not os.path.isdir(tf_record_file_path):
        os.makedirs(tf_record_file_path)

    date_ls = DateUtil.get_every_date(start_date, end_date)

    for date in date_ls:
        print(date)
        tfRecorder.transfer_single_feature_file_2_tfRecord(raw_feature_file % date, tf_record_file % date,
                                                           data_info_csv_path, column_names,
                                                           label_name, need_features_cols,
                                                           var_length_cols=var_length_cols,
                                                           col_preprocess_func=col_preprocess_func)


def main():
    import sys
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    raw_feature_file_path = os.path.join(FEATURE_PATH, "wechat_record_feature")
    tf_record_file_path = os.path.join(TF_RECORD_PATH, "wechat_record_feature")
    from config.global_config_text import FILTER_NULL_WECHAT_SAMPLE
    if FILTER_NULL_WECHAT_SAMPLE:
        tf_record_file_path = tf_record_file_path + "_remove_absent"
    transfer_wechat_feature_2_tfRecord(start_date, end_date, raw_feature_file_path, tf_record_file_path)


if __name__ == '__main__':
    main()
