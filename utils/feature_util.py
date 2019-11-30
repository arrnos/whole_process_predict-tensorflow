# -*- coding: utf-8 -*-

"""
@author: liutao
@file: feature_util.py
@time: 2019/10/9 17:28
"""

import os
import codecs
from tensorflow import feature_column
from utils.date_util import DateUtil
from utils.TFRecord_util import get_tf_type
from feature_infos import DICT_FEATURE_DTYPE


def _numeric_feature(feature_name):

    return feature_column.numeric_column(feature_name)


def _bucketized_feature(feature_name, feature_boundaries):

    numeric_feature = _numeric_feature(feature_name)
    return feature_column.bucketized_column(numeric_feature, boundaries=feature_boundaries)


def _categorical_feature(feature_name, vocabulary_list):

    categorical_feature = feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary_list)
    return feature_column.indicator_column(categorical_feature)


def _embedding_feature(feature_name, vocabulary_list, dimenssion):

    categorical_feature = _categorical_feature(feature_name, vocabulary_list)
    embedding_feature = feature_column.embedding_column(categorical_feature, dimenssion=dimenssion)
    return embedding_feature


def get_vocabulary_list(vocab_file_dir, feature_name, start_date, end_date):
    
    vocab_set = set()
    file_prefix = os.path.join(vocab_file_dir, feature_name + "_vocabulary_")
    date_ls = DateUtil.get_every_date(start_date, end_date)
    for tmp_date in date_ls:
        with codecs.open(file_prefix + tmp_date, "r", "utf-8") as fin:
            for line in fin:
                vocab_set.add(line.strip().encode("utf-8"))
    vocab_set.add("0".encode("utf-8"))
    
    return list(vocab_set)


def get_MinMaxValue_dict(MinMaxValue_file_dir, start_date, end_date):
    
    MinMaxValue_dict = dict()
    file_prefix = os.path.join(MinMaxValue_file_dir, "MinMaxValue_file_")
    date_ls = DateUtil.get_every_date(start_date, end_date)
    for tmp_date in date_ls:
        with codecs.open(file_prefix + tmp_date, "r", "utf-8") as fin:
            for line in fin:
                arr = line.strip().split("\t")
                feature_name = arr[0].strip()
                minValue = float(arr[1].strip())
                maxValue = float(arr[2].strip())
                if feature_name not in MinMaxValue_dict:
                    MinMaxValue_dict[feature_name] = [minValue, maxValue]
                    continue
                if minValue < MinMaxValue_dict[feature_name][0]:
                    MinMaxValue_dict[feature_name][0] = minValue
                if maxValue > MinMaxValue_dict[feature_name][1]:
                    MinMaxValue_dict[feature_name][1] = maxValue

    return MinMaxValue_dict