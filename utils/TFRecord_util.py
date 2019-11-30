# -*- coding: utf-8 -*-

"""
@author: liutao
@file: TFRecord_util.py
@time: 2019/10/9 17:25
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Iterable


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode("utf-8") if isinstance(value, str) else value]))


def get_tf_feature(value):
    if isinstance(value, int) or isinstance(value, np.int64):
        return _int64_feature(value)
    elif isinstance(value, float) or isinstance(value, np.float32):
        return _float_feature(value)
    elif isinstance(value, str) or isinstance(value, bytes):
        return _bytes_feature(value)
    else:
        raise TypeError("value type is %s! but expected type is int, float, or str." % type(value))
    
    
def get_tf_type(raw_type):
    if raw_type == np.int64:
        return tf.int64
    elif raw_type == np.float32:
        return tf.float32
    else:
        return tf.string


def dump_TFRecord_file(data, create_tf_example, file_path):
    if isinstance(data, pd.DataFrame):
        writer = tf.io.TFRecordWriter(file_path)
        for idx, row in data.iterrows():
            tf_example = create_tf_example(row)
            writer.write(tf_example.SerializeToString())
        writer.close()

    elif isinstance(data, Iterable):
        writer = tf.io.TFRecordWriter(file_path)
        for row in data:
            tf_example = create_tf_example(row)
            writer.write(tf_example.SerializeToString())
        writer.close()

    else:
        raise TypeError("Required type is pd.DataFrame or Iterable.")