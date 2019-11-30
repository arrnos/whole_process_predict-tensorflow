# -*- coding: utf-8 -*-

"""
@author: liutao
@file: transform_feature_2_TFRecord.py
@time: 2019/10/11 11:16
"""

import os
import codecs
from feature_infos import *
from config.global_config import PARENT_DATA_DIR, VOCABULARY_DIR, MinMaxValue_DIR
from utils.TFRecord_util import *
from log.get_logger import LOG


def get_feature_index_and_value(name_value):
    index = -1

    if "oppor_source" in name_value:
        arr = name_value.strip().split("oppor_source_")
        value = arr[1].strip().split(":")[0].strip()
        index = DICT_FEATURE_INDEX["oppor_source"]
        return index, value

    if "config_value" in name_value:
        arr = name_value.strip().split("config_value_")
        value = arr[1].strip().split(":")[0].strip()
        index = DICT_FEATURE_INDEX["config_value"]
        return index, value

    arr = name_value.strip().split(":")
    name = arr[0].strip()
    value = arr[1].strip()
    if name in DICT_FEATURE_INDEX:
        index = DICT_FEATURE_INDEX[name]
    else:
        name_arr = name.split("_")
        new_name = "_".join(name_arr[:-1])
        if new_name in DICT_FEATURE_INDEX:
            index = DICT_FEATURE_INDEX[new_name]
            value = name_arr[-1].strip()

    return index, value


def update_MinMaxValue_dict(MinMaxValue_dict, index, value):
    feature_name = FEATURE_INFOS[index][0]
    feature_type = FEATURE_INFOS[index][1]
    if feature_type == NUMERIC:
        if feature_name not in MinMaxValue_dict:
            MinMaxValue_dict[feature_name] = [float(value), float(value)]
        if float(value) < MinMaxValue_dict[feature_name][0]:
            MinMaxValue_dict[feature_name][0] = float(value)
        if float(value) > MinMaxValue_dict[feature_name][1]:
            MinMaxValue_dict[feature_name][1] = float(value)


def dump_MinMaxValue(date_str, MinMaxValue_dict):

    MinMaxValue_file = os.path.join(MinMaxValue_DIR, "MinMaxValue_file_%s" % date_str)

    with codecs.open(MinMaxValue_file, "w", "utf-8") as fout:
        for feature_name, MinMaxValue in MinMaxValue_dict.items():
            fout.write(feature_name + "\t" + "%.9f\t%.9f" % (MinMaxValue[0], MinMaxValue[1]) + "\n")


def update_vocabulary_dict(vocabulary_dict, index, value):

    feature_name = FEATURE_INFOS[index][0]
    feature_type = FEATURE_INFOS[index][1]
    if feature_type == CATEGORICAL:
        if feature_name not in vocabulary_dict:
            vocabulary_dict[feature_name] = set()
        vocabulary_dict[feature_name].add(value)


def dump_vocabulary(date_str, vocabulary_dict):

    for feature_name, vocab_set in vocabulary_dict.items():
        vocabulary_file = os.path.join(VOCABULARY_DIR, "%s_vocabulary" % feature_name, "%s_vocabulary_%s" % (feature_name, date_str))
        with codecs.open(vocabulary_file, "w", "utf-8") as fout:
            for tmp_value in vocab_set:
                fout.write(tmp_value + "\n")


def transform_feature_2_csv(date_str, input_file, output_file):
    feature_nums = len(DICT_FEATURE_INDEX)

    vocabulary_dict = dict()
    MinMaxValue_dict = dict()

    with codecs.open(input_file, "r", "utf-8") as fin, codecs.open(output_file, "w", "utf-8") as fout:
        for line in fin:
            arr = line.strip().split(" ")
            label = arr[0].strip()
            result_ls = ['0'] * feature_nums
            result_ls[0] = label
            for name_value in arr[1:]:
                index, value = get_feature_index_and_value(name_value)
                if index == -1:
                    continue
                update_vocabulary_dict(vocabulary_dict, index, value)
                update_MinMaxValue_dict(MinMaxValue_dict, index, value)
                if index in [137, 138]:
                    result_ls[index] += " " + value
                else:
                    result_ls[index] = value
            result_str = ",".join(result_ls)
            fout.write(result_str.strip() + "\n")

    dump_vocabulary(date_str, vocabulary_dict)
    dump_MinMaxValue(date_str, MinMaxValue_dict)


def transform_csv_feature_2_TFRecord(input_file, output_file):
    data = pd.read_csv(input_file, header=None, names=FEATURE_NAMES, dtype=DICT_FEATURE_DTYPE, encoding='utf-8')
    dump_TFRecord_file(data, create_tf_example, output_file)


def transform_one_day_feature_2_TFRecord(date_str):
    # merged_feature = os.path.join(PARENT_DATA_DIR, "feature_file/merged_feature", "merged_feature_%s" % date_str)
    merged_feature = os.path.join("/home/yanxin/project_data/whole_process_extend_sample", "feature_file/merged_feature", "merged_feature_%s" % date_str)
    csv_feature = os.path.join(PARENT_DATA_DIR, "feature_file/csv_feature", "csv_feature_%s" % date_str)
    TFRecord_feature = os.path.join(PARENT_DATA_DIR, "feature_file/TFRecord_feature", "TFRecord_feature_%s" % date_str)
    
    LOG.info("Start to transform feature 2 csv...")
    transform_feature_2_csv(date_str, merged_feature, csv_feature)
    LOG.info("Start to transform csv feature 2 TFRecord...")
    transform_csv_feature_2_TFRecord(csv_feature, TFRecord_feature)


def create_tf_example(sample):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        key: get_tf_feature(sample[key]) for key in FEATURE_NAMES
    }))

    return tf_example


def parse_function(tf_example):
    feature_description = {
        key: tf.io.FixedLenFeature([], get_tf_type(raw_type)) for key, raw_type in DICT_FEATURE_DTYPE.items()
    }

    raw_example = tf.io.parse_single_example(tf_example, feature_description)
    features, label = raw_example, raw_example.pop("label")
    features["audio"] = tf.strings.split(features["audio"], sep=" ")
    features["wechat"] = tf.strings.split(features["wechat"], sep=" ")
    return features, label


# def parse_function(tf_example):
#     feature_description = {
#         "image": tf.io.FixedLenFeature([], tf.string),
#         "label": tf.io.FixedLenFeature([], tf.int64),
#     }
#
#     raw_example = tf.io.parse_single_example(tf_example, feature_description)
#     image = tf.io.decode_raw(raw_example["image"], tf.float32)
#     features = {"image": image}
#     label = raw_example["label"]
#     return features, label


def main():
    # train_file_names = ["../data/mnist/train_data.TFRecord"]
    # valid_file_names = ["../data/mnist/test_data.TFRecord"]
    # test_file_names = ["../data/mnist/test_data.TFRecord"]
    # merged_feature = "merged_feature_20191010"
    # csv_feature_file = "csv_feature_file_20191010"
    # TFRecord_file = "TFRecord_file_20191010"
    # transform_feature_2_csv("20191010", merged_feature, csv_feature_file)
    # transform_csv_feature_2_TFRecord(csv_feature_file, TFRecord_file)
    TFRecord_feature = os.path.join(PARENT_DATA_DIR, "feature_file/TFRecord_feature", "TFRecord_feature_20190725")
    # dataset = tf.data.TFRecordDataset([TFRecord_feature])
    # date_str = u"20190823"
    # merged_feature = "../tmp.txt"
    # csv_feature_file = "../sample.csv"
    # TFRecord_file = "../TFRecord_file"
    # transform_feature_2_csv(merged_feature, csv_feature_file)
    # transform_csv_feature_2_TFRecord(csv_feature_file, TFRecord_file)
    # feature_columns = [tf.feature_column.numeric_column("image", shape=(784,))]
    from model.DNN import get_padded_shapes_and_values
    dataset = tf.data.TFRecordDataset([TFRecord_feature]).map(parse_function)
    for features, label in dataset.take(1):
        for key, value in features.items():
            print(key, value.numpy())
        print(label)
    # model = tf.keras.Sequential([
    #     tf.keras.layers.DenseFeatures(feature_columns),
    #     tf.keras.layers.Dense(10, activation="relu"),
    #     tf.keras.layers.Dense(10, activation="sigmoid")
    # ])
    # run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    # run_metadata = tf.compat.v1.RunMetadata()
    # model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #               metrics=["accuracy"],
    #               options=run_options,
    #               run_metadata=run_metadata)
    # model.fit(dataset, epochs=10)
    # model.evaluate(dataset)
    # from tensorflow.python.client import timeline
    # tl = timeline.Timeline(run_metadata.step_stats)
    # ctf = tl.generate_chrome_trace_format()
    # with codecs.open('timeline.json', 'w') as f:
    #     f.write(ctf)




if __name__ == '__main__':
    main()
