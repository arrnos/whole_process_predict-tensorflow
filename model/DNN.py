# -*- coding: utf-8 -*-

"""
@author: liutao
@file: DNN.py
@time: 2019/10/10 16:21
"""

import os
import itertools
import numpy as np
import tensorflow as tf
from config.global_config import VOCABULARY_DIR, PARENT_DATA_DIR, MinMaxValue_DIR
from feature_infos import *
from utils.TFRecord_util import *
from utils.feature_util import get_vocabulary_list, get_MinMaxValue_dict

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def transform_MinMaxValue_2_tensor(MinMaxValue_dict, feature_dtype_dict):
    MinMaxValue_tensor_dict = dict()
    for feature_name, MinMaxValue in MinMaxValue_dict.items():
        if feature_name != "opp_create_obs_interval_minutes":
            continue
        raw_type = feature_dtype_dict[feature_name]
        if raw_type == np.int64:
            minValue = int(MinMaxValue[0])
            maxValue = int(MinMaxValue[1])
        elif raw_type == np.float32:
            minValue = float(MinMaxValue[0])
            maxValue = float(MinMaxValue[1])
        else:
            raise TypeError("transform_MinMaxValue_2_tensor occurred type error...")
        rangeValue = maxValue - minValue
        tf_type = get_tf_type(raw_type)
        MinMaxValue_tensor_dict[feature_name] = []
        MinMaxValue_tensor_dict[feature_name].append(tf.constant(minValue, dtype=tf_type))
        MinMaxValue_tensor_dict[feature_name].append(tf.constant(maxValue, dtype=tf_type))
        MinMaxValue_tensor_dict[feature_name].append(tf.constant(rangeValue, dtype=tf_type))

    return MinMaxValue_tensor_dict


def get_discretization_set():
    log_set = set()
    for feature_name, feature_type in DICT_FEATURE_TYPE.items():
        if feature_type == CATEGORICAL or feature_name == "label":
            continue
        if "eff_num" in feature_name:
            continue
        if "ratio" in feature_name:
            continue
        log_set.add(feature_name)

    return log_set


def create_tf_example(raw_example):
    image = raw_example[0]
    label = raw_example[1]
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image": get_tf_feature(image.tostring()),
        "label": get_tf_feature(label)
    }))

    return tf_example


def get_sub_str_list(raw_str, sep, length):
    str_list = tf.strings.split(raw_str, sep=sep)
    result = tf.cond(tf.size(str_list) > length, lambda: tf.slice(str_list, [0], [length]), lambda: str_list)
    return result


feature_description = {
    key: tf.io.FixedLenFeature([], get_tf_type(raw_type)) for key, raw_type in DICT_FEATURE_DTYPE.items()
}
log_set = get_discretization_set()


# def parse_function(tf_example, MinMaxValue_tensor_dict):
#     raw_example = tf.io.parse_single_example(tf_example, feature_description)
#     features, label = raw_example, raw_example.pop("label")
#     for feature_name, MinMaxValue in MinMaxValue_tensor_dict.items():
#         minValue = MinMaxValue[0]
#         rangeValue = MinMaxValue[2]
#         features[feature_name] = tf.divide(tf.subtract(features[feature_name], minValue), rangeValue)
#     del features["audio"]
#     del features["wechat"]
#     return features, label

# def parse_function(tf_example, MinMaxValue_tensor_dict):
#     raw_example = tf.io.parse_single_example(tf_example, feature_description)
#     features = {
#         "province_id": raw_example["province_id"],
#         "city_id": raw_example["city_id"],
#         "site_id": raw_example["site_id"],
#         "legion_id": raw_example["legion_id"],
#         "quantum_id": raw_example["quantum_id"],
#         "site_source": raw_example["site_source"],
#         "first_proj_id": raw_example["first_proj_id"],
#         "oppor_source": raw_example["oppor_source"],
#         "config_value": raw_example["config_value"],
#         "opp_create_obs_interval_minutes": raw_example["opp_create_obs_interval_minutes"],
#     }
#     label = raw_example.pop("label")
#     for feature_name, MinMaxValue in MinMaxValue_tensor_dict.items():
#         minValue = MinMaxValue[0]
#         rangeValue = MinMaxValue[2]
#         features[feature_name] = tf.divide(tf.subtract(features[feature_name], minValue), rangeValue)
#     return features, label


def parse_function(tf_example):
    raw_example = tf.io.parse_single_example(tf_example, feature_description)
    features, label = raw_example, raw_example.pop("label")
    del features["audio"]
    del features["wechat"]
    for feature_name in log_set:
        features[feature_name] = tf.math.log1p(tf.dtypes.cast(features[feature_name], dtype=tf.float32))
    return features, label


def mnist_parse_function(tf_example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    raw_example = tf.io.parse_single_example(tf_example, feature_description)
    image = tf.io.decode_raw(raw_example["image"], tf.float32)
    features = {"image": image, "text": image}
    label = raw_example["label"]
    return features, label


def get_feature_columns(start_date, end_date):
    feature_columns = []
    for feature_name, feature_type in DICT_FEATURE_TYPE.items():
        if feature_name == "label" or feature_name == "audio" or feature_name == "wechat":
            continue
        if feature_type == NUMERIC:
            numeric_column = tf.feature_column.numeric_column(feature_name)
            if "count" in feature_name:
                bucket_column = tf.feature_column.bucketized_column(numeric_column,
                                                                    boundaries=[0.01, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                embedding_column = tf.feature_column.embedding_column(bucket_column, dimension=4)
                feature_columns.append(embedding_column)
            elif "ratio" in feature_name:
                bucket_column = tf.feature_column.bucketized_column(numeric_column,
                                                                    boundaries=[0.01, 0.25, 0.5, 0.75, 1.0, 5.0, 10.0])
                embedding_column = tf.feature_column.embedding_column(bucket_column, dimension=4)
                feature_columns.append(embedding_column)
            else:
                bucket_column = tf.feature_column.bucketized_column(numeric_column,
                                                                    boundaries=[0.01, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                                                12, 13, 14, 15, 16])
                embedding_column = tf.feature_column.embedding_column(bucket_column, dimension=4)
                feature_columns.append(embedding_column)
        elif feature_type == CATEGORICAL:
            vocab_file_dir = os.path.join(VOCABULARY_DIR, "%s_vocabulary" % feature_name)
            vocab_list = get_vocabulary_list(vocab_file_dir, feature_name, start_date, end_date)
            categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                                                           vocabulary_list=vocab_list)
            embedding_column = tf.feature_column.embedding_column(categorical_column, dimension=12)
            feature_columns.append(embedding_column)

    return feature_columns


def get_padded_shapes_and_values():
    padded_shapes = dict()
    padded_values = dict()
    for feature_name, feature_dtype in DICT_FEATURE_DTYPE.items():
        if feature_name == "label":
            continue
        if feature_name == "audio" or feature_name == "wechat":
            padded_shapes[feature_name] = tf.TensorShape([100])
            padded_values[feature_name] = b""
            continue
        if feature_dtype == np.int64:
            padded_shapes[feature_name] = tf.TensorShape([])
            padded_values[feature_name] = 0
        elif feature_dtype == np.float32:
            padded_shapes[feature_name] = tf.TensorShape([])
            padded_values[feature_name] = 0.0
        elif feature_dtype == str:
            padded_shapes[feature_name] = tf.TensorShape([])
            padded_values[feature_name] = b""

    return padded_shapes, padded_values


def get_TFRecord_file(start_date, end_date):
    TFRecord_file_dir = os.path.join(PARENT_DATA_DIR, "feature_file/TFRecord_feature")
    start_file_name = "TFRecord_feature_%s" % start_date
    end_file_name = "TFRecord_feature_%s" % end_date

    file_names = [os.path.join(TFRecord_file_dir, tmp_file) for tmp_file in os.listdir(TFRecord_file_dir) if
                  start_file_name <= tmp_file <= end_file_name]
    return file_names


class DNN(tf.keras.Model):
    def __init__(self, feature_columns):
        super(DNN, self).__init__()
        # self.image_tensor = tf.keras.layers.DenseFeatures([tf.feature_column.numeric_column("image", shape=(784,))])
        # self.text_tensor = tf.keras.layers.DenseFeatures([tf.feature_column.numeric_column("text", shape=(784,))])
        self.dense_tensor = tf.keras.layers.DenseFeatures(feature_columns)
        self.full_connec = tf.keras.layers.Dense(16)
        self.classifier = tf.keras.layers.Dense(1)

    def call(self, features):
        output = self.dense_tensor(features)
        output = self.full_connec(output)
        output = self.classifier(output)

        # image_feature = {"image": features.pop("image")}
        # text_feature = {"text": features.pop("text")}
        #
        # output = self.concat([self.image_tensor(image_feature), self.text_tensor(text_feature)])
        # output = self.full_connec(output)
        # output = self.classifier(output)

        return output


def mnist_main():
    # train_images_file = "../data/mnist/train_images.npy"
    # train_labels_file = "../data/mnist/train_labels.npy"
    # test_images_file = "../data/mnist/test_images.npy"
    # test_labels_file = "../data/mnist/test_labels.npy"
    # train_images = np.load(train_images_file)
    # train_labels = np.load(train_labels_file)
    # test_images = np.load(test_images_file)
    # test_labels = np.load(test_labels_file)
    train_file_names = ["../data/mnist/train_data.TFRecord"]
    test_file_names = ["../data/mnist/test_data.TFRecord"]
    feature_columns = []
    feature_columns.append(tf.feature_column.numeric_column("image", shape=(1, 784)))
    model = tf.keras.Sequential([
        tf.keras.layers.DenseFeatures(feature_columns),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="sigmoid")
    ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    train_dataset = tf.data.TFRecordDataset(train_file_names).map(mnist_parse_function).batch(256)
    test_dataset = tf.data.TFRecordDataset(test_file_names).map(mnist_parse_function).batch(256)
    model.fit(train_dataset)
    print(model.evaluate(test_dataset))
    print(list(itertools.islice(model.predict(test_dataset, 10))))


def main():
    import sys
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    valid_start_date = sys.argv[3]
    valid_end_date = sys.argv[4]
    test_start_date = sys.argv[5]
    test_end_date = sys.argv[6]
    train_file_names = get_TFRecord_file(start_date, end_date)
    valid_file_names = get_TFRecord_file(valid_start_date, valid_end_date)
    test_file_names = get_TFRecord_file(test_start_date, test_end_date)
    feature_columns = get_feature_columns(start_date, end_date)
    # MinMaxValue_dict = get_MinMaxValue_dict(MinMaxValue_DIR, start_date, end_date)
    # MinMaxValue_tensor_dict = transform_MinMaxValue_2_tensor(MinMaxValue_dict, DICT_FEATURE_DTYPE)
    model = tf.keras.Sequential([
        tf.keras.layers.DenseFeatures(feature_columns),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ]
    )
    # padded_shapes, padded_values = get_padded_shapes_and_values()
    train_dataset = tf.data.TFRecordDataset(train_file_names).map(
        parse_function, num_parallel_calls=8).shuffle(10240).batch(
        1024).prefetch(5)
    valid_dataset = tf.data.TFRecordDataset(valid_file_names).map(
        parse_function, num_parallel_calls=8).batch(
        1024).prefetch(5)
    test_dataset = tf.data.TFRecordDataset(test_file_names).map(
        parse_function, num_parallel_calls=8).batch(
        1024).prefetch(5)
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    for features, label in train_dataset.take(1):
        for key, value in features.items():
            print(key, value.shape)
        print(label)
        print(feature_layer(features))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC()])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, mode="max")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0)
    history = model.fit(train_dataset, epochs=20, callbacks=[early_stop, reduce_lr], validation_data=valid_dataset,
                        class_weight={0: 1., 1: 50.})
    print(history)
    print(model.evaluate(test_dataset))
    print(list(itertools.islice(model.predict(test_dataset), 10)))
    model.save("whole_model", save_format="tf")


if __name__ == '__main__':
    main()
