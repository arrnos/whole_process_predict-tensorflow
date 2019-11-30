# -*- coding: utf-8 -*-

"""
@author: liutao
@file: DNN.py
@time: 2019/10/10 16:21
"""

import itertools
import os

from config.feature_config.feature_column_config import BUCKTE_EMBEDDING_SIZE, SPARSE_EMBEDDING_SIZE
from config.feature_config.feature_column_config import InteractionColumns, LinnerColumns, InteractionColumns_pool, \
    DNNColumns, DNNColumns_pool
from config.feature_config.feature_infos import *
from config.feature_config.feature_infos import FEATURE_INPUT_DICT
from config.global_config import PARENT_DATA_DIR
from utils.TFRecord_util import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

feature_description = {
    key: tf.io.FixedLenFeature([], get_tf_type(raw_type)) for key, raw_type in zip(FEATURE_NAMES, FEATURE_DTYPES)
}


def parse_function(tf_example):
    raw_example = tf.io.parse_single_example(tf_example, feature_description)
    features, label = raw_example, raw_example.pop("label")
    return features, label


def get_TFRecord_file(start_date, end_date):
    TFRecord_file_dir = os.path.join(PARENT_DATA_DIR, "feature_file/TFRecord_feature")
    start_file_name = "TFRecord_feature_%s" % start_date
    end_file_name = "TFRecord_feature_%s" % end_date

    file_names = [os.path.join(TFRecord_file_dir, tmp_file) for tmp_file in os.listdir(TFRecord_file_dir) if
                  start_file_name <= tmp_file <= end_file_name]
    return file_names


def check_dataset_read(train_dataset):
    print("LinnerColumns")
    feature_layer = tf.keras.layers.DenseFeatures(LinnerColumns)
    for features, label in train_dataset.take(1):
        print(feature_layer(features))
    # ====
    # print("DNNColumns")
    # feature_layer = tf.keras.layers.DenseFeatures(DNNColumns_pool)
    # for features, label in train_dataset.take(1):
    #     print(feature_layer(features))
    # ====
    # print("InteractionColumns")
    # feature_layer = tf.keras.layers.DenseFeatures(InteractionColumns_pool)
    # for features, label in train_dataset.take(1):
    #     print(feature_layer(features))


def main():
    import sys
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    valid_start_date = sys.argv[3]
    valid_end_date = sys.argv[4]
    test_start_date = sys.argv[5]
    test_end_date = sys.argv[6]
    batch_size = 512
    train_file_names = get_TFRecord_file(start_date, end_date)
    valid_file_names = get_TFRecord_file(valid_start_date, valid_end_date)
    test_file_names = get_TFRecord_file(test_start_date, test_end_date)

    train_dataset = tf.data.TFRecordDataset(train_file_names).map(
        parse_function, num_parallel_calls=8).filter(
        lambda x, y: True if tf.reduce_sum(y) != 0 else np.random.random() < 0.3).shuffle(
        10240).batch(
        batch_size).prefetch(5)
    valid_dataset = tf.data.TFRecordDataset(valid_file_names).map(
        parse_function, num_parallel_calls=8).batch(
        batch_size).prefetch(5)
    test_dataset = tf.data.TFRecordDataset(test_file_names).map(
        parse_function, num_parallel_calls=8).batch(
        batch_size).prefetch(5)

    check_dataset_read(valid_dataset)

    model = dfm_model()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC()])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=4, mode="max")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0, verbose=1)
    history = model.fit(train_dataset, epochs=3, callbacks=[early_stop, reduce_lr], validation_data=valid_dataset,
                        class_weight={0: 1., 1: 200.})
    print(history.history)
    # print(model.evaluate(test_dataset))
    # print(list(itertools.islice(model.predict(test_dataset), 10)))
    # model.save("whole_model", save_format="tf")


def dfm_model(dnn_units=(128, 64), drop_ratio=0.5, use_liner=True, use_fm=True, use_dnn=True):
    def get_merged_embedding(input_feature):
        # # shape(batch_size, column_num * embedding_size)
        no_need_pool_emb = tf.keras.layers.DenseFeatures(InteractionColumns_pool[0])(input_feature)
        assert BUCKTE_EMBEDDING_SIZE == SPARSE_EMBEDDING_SIZE
        column_num, dimension = len(InteractionColumns_pool[0]), SPARSE_EMBEDDING_SIZE
        no_need_pool_vals = tf.reshape(no_need_pool_emb, (-1, column_num, dimension), "interaction_embeddings")
        pool_list = []
        for x in InteractionColumns_pool[1:]:
            emb = tf.keras.layers.DenseFeatures(x)(input_feature)
            column_num, dimension = len(x), SPARSE_EMBEDDING_SIZE
            vals = tf.reduce_mean(tf.reshape(emb, (-1, column_num, dimension)), 1, keepdims=True)
            pool_list.append(vals)
        concat_val = tf.concat(pool_list + [no_need_pool_vals], axis=1, name="concat_merge")
        return concat_val

    def fm_logit(concat_embedding):
        # sum-square-part
        summed_val = tf.reduce_sum(concat_embedding, 1)
        summed_square_val = tf.square(summed_val)

        # squre-sum-part
        squared_val = tf.square(concat_embedding)
        squared_sum_val = tf.reduce_sum(squared_val, 1)

        # second order
        logit = tf.reduce_sum(0.5 * tf.subtract(summed_square_val, squared_sum_val), -1)
        return logit

    def liner_logit(input_feature):
        input = tf.keras.layers.DenseFeatures(LinnerColumns)(input_feature)
        logit = tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input)

        return logit

    def dnn_logit(concat_embedding, dnn_units, drop_ratio):
        output = tf.keras.layers.Flatten(name="embedding_flatten")(concat_embedding)
        for unit in dnn_units:
            output = tf.keras.layers.Dense(unit, kernel_regularizer=tf.keras.regularizers.l2(0.005),
                                           bias_regularizer=tf.keras.regularizers.l2(0.005))(output)
            if drop_ratio:
                output = tf.keras.layers.Dropout(drop_ratio)(output)

        logit = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.005), use_bias=False)(output)

        return logit

    input_feature = FEATURE_INPUT_DICT

    logit_ls = []
    concat_embedding = get_merged_embedding(input_feature)
    if use_dnn:
        dnn_logits = dnn_logit(concat_embedding, dnn_units, drop_ratio)
        logit_ls.append(dnn_logits)
    if use_fm:
        fm_logits = fm_logit(concat_embedding)
        logit_ls.append(fm_logits)
    if use_liner:
        liner_logits = liner_logit(input_feature)
        logit_ls.append(liner_logits)

    assert logit_ls
    if len(logit_ls) > 1:
        output = tf.keras.layers.add(logit_ls)
    else:
        output = logit_ls[0]

    pred = tf.keras.layers.Dense(1, activation="sigmoid")(output)

    model = tf.keras.Model(inputs=input_feature, outputs=pred)
    return model


if __name__ == '__main__':
    main()
