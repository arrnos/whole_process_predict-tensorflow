#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: make_bench_mark_data.py
@time: 2019/10/24
"""
from config.global_config_text import *
from utils.date_util import DateUtil
from utils.tf_recorder_util import TFRecorder
import codecs

chat_num_ls = [1, 2, 5, 10, 20, 40, 60, 80]
source_file_folder_name = "wechat_record_feature"
raw_feature_path = os.path.join(FEATURE_PATH, source_file_folder_name)
bench_mark_text_file_path = os.path.join(FEATURE_PATH, "%s_bench_mark" % source_file_folder_name)
tfRecorder = TFRecorder()


def gen_multi_day_bench_mark_data(start_date, end_date):
    bench_mark_text_file_tmp = os.path.join(bench_mark_text_file_path, "total_chat_num_%s", "total_chat_num_%s_%s")
    for date in DateUtil.get_every_date(start_date, end_date):
        print(date)
        # 按chat_num分组
        source_file_path = os.path.join(raw_feature_path, "wechat_record_feature_%s" % date)
        chat_group_dict = {x: [] for x in chat_num_ls}

        with codecs.open(source_file_path, 'r', 'utf-8') as fin:
            for line in fin:
                arr = line.strip().split("\t")
                column_names = ["label", "opp_id", "create_time", "hist_student_chat_num",
                                "hist_teacher_chat_num", "hist_total_chat_num", "wechat_record"]
                tmp_dict = {key: value for key, value in zip(column_names, arr)}
                total_student_chat_num = int(tmp_dict["hist_student_chat_num"])
                if total_student_chat_num in chat_group_dict:
                    chat_group_dict[total_student_chat_num].append(line.strip())

        for chat_num, line_ls in chat_group_dict.items():
            if not line_ls:
                continue

            bench_mark_text_file = bench_mark_text_file_tmp % (chat_num, chat_num, date)
            tmp_text_folder_path = os.path.dirname(bench_mark_text_file)
            tmp_tf_record_folder_path = tmp_text_folder_path.replace("feature_file", "tf_record")

            if not os.path.isdir(tmp_text_folder_path):
                os.makedirs(tmp_text_folder_path)

            with codecs.open(bench_mark_text_file, "w", "utf-8") as fout:
                for line in line_ls:
                    fout.write(line + "\n")
            # 将转换好的text bench mark 文本文件 ， 转换为tfRecord文件
            tfRecorder.transfer_wechat_feature_2_tfRecord_default(date, date, tmp_text_folder_path,
                                                                  tmp_tf_record_folder_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-start_date", type=str, default="20190803")
    parser.add_argument("-end_date", type=str, default="20190803")
    args = parser.parse_args()
    gen_multi_day_bench_mark_data(args.start_date, args.end_date)


if __name__ == '__main__':
    main()
