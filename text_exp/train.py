import argparse
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config.global_config_text import *
from utils import text_transform_util
from text_exp.evaluate import test_benchmark_and_all
from model.TextCnn import TextCnn
from text_exp.dataset_read_config import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
np.set_printoptions(threshold=np.inf)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[1],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def training(train_dataset, valid_dataset, vocab_size, epochs, model_saved_path, log_path):
    model = TextCnn(args.feature_size, args.embedding_size, vocab_size, args.filter_num,
                    args.filter_list, args.drop_out_ratio)
    model.compile(tf.keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC()])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_auc', mode="max", patience=5),
        keras.callbacks.TensorBoard(log_dir=log_path),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1, mode='min',
                                          epsilon=0.0001, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(model_saved_path + "_epoch-{epoch:02d}_val_auc-{val_auc:.6f}.hdf5",
                                        monitor='val_auc', mode="max",
                                        verbose=1, save_best_only=True)
    ]

    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=callbacks)

    print("\nLog path:", log_path)
    print("\nSave model:", model_saved_path)
    keras.models.save_model(model, model_saved_path, save_format='h5')

    print(history.history)
    return model


def prepare_dataset(date_ls, train_tf_record_folder_name, valid_tf_record_folder_name):
    assert len(date_ls) == 4
    assert date_ls[0] <= date_ls[1] <= date_ls[2] <= date_ls[3]
    print("加载数据集...\n训练集：%s-%s\n验证集：%s-%s" % (date_ls[0], date_ls[1], date_ls[2], date_ls[3]))

    train_tf_record_path = os.path.join(TF_RECORD_PATH, train_tf_record_folder_name)
    valid_tf_record_path = os.path.join(TF_RECORD_PATH, valid_tf_record_folder_name)

    train_dataset = tfRecorder.get_dataset_from_path(train_tf_record_path, feature_ls, label_name=label_name,
                                                     start_date=date_ls[0], end_date=date_ls[1],
                                                     batch_size=args.batch_size, padding=padding)
    valid_dataset = tfRecorder.get_dataset_from_path(valid_tf_record_path, feature_ls, label_name=label_name,
                                                     start_date=date_ls[2], end_date=date_ls[3],
                                                     batch_size=args.batch_size, padding=padding)
    return train_dataset, valid_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="textCnn model..")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("-epochs", "--epochs", type=int, default=3, help="epochs")
    parser.add_argument("-feature_size", "--feature_size", type=int, default=SEQUENCE_MAX_LEN, help="feature_size")
    parser.add_argument("-embedding_size", "--embedding_size", type=int, default=32, help="embedding_size")
    parser.add_argument("-filter_num", "--filter_num", type=int, default=16, help="filter_num")
    parser.add_argument("-filter_list", "--filter_list", type=str, default="3,4,5,6", help="filter_list")
    parser.add_argument("-drop_out_ratio", "--drop_out_ratio", type=float, default=0.5, help="drop_out_ratio")
    parser.add_argument("-result_dir", "--result_dir", type=str, default=os.path.join(PROJECT_DATA_DIR, "result"),
                        help="result_dir")

    parser.add_argument("-train_start_date", "--train_start_date", type=str, default="20190725",
                        help="train_start_date")
    parser.add_argument("-train_end_date", "--train_end_date", type=str, default="20190813", help="train_end_date")
    parser.add_argument("-valid_start_date", "--valid_start_date", type=str, default="20190821",
                        help="valid_start_date")
    parser.add_argument("-valid_end_date", "--valid_end_date", type=str, default="20190823", help="valid_end_date")
    parser.add_argument("-test_start_date", "--test_start_date", type=str, default="20190821", help="test_start_date")
    parser.add_argument("-test_end_date", "--test_end_date", type=str, default="20190823", help="test_end_date")

    parser.add_argument("-train_tf_record_folder_name", "--train_tf_record_folder_name", type=str,
                        default="wechat_record_feature_remove_absent", help="train_tf_record_folder_name")
    parser.add_argument("-valid_tf_record_folder_name", "--valid_tf_record_folder_name", type=str,
                        default="wechat_record_feature_remove_absent", help="valid_tf_record_folder_name")
    parser.add_argument("-test_tf_record_folder_name", "--test_tf_record_folder_name", type=str,
                        default="wechat_record_feature_bench_mark", help="test_tf_record_folder_name")
    parser.add_argument("-pos_sample_weight", "--pos_sample_weight", type=float,
                        default=40.0, help="pos_sample_weight")

    parser.add_argument("-is_test", "--is_test", type=bool, default=False, help="is_test")

    args = parser.parse_args()
    print("\nArgument:", args, "\n")

    # Prepare 结果输出目录
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H-%M")
    log_path = os.path.join(args.result_dir, timestamp, "logs")
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    print("\nPrepare train data and valid data..")
    date_ls = [args.train_start_date, args.train_end_date, args.valid_start_date, args.valid_end_date]
    train_dataset, valid_dataset = prepare_dataset(date_ls, args.train_tf_record_folder_name,
                                                   args.valid_tf_record_folder_name)

    print("\nTraining")
    model_saved_path = os.path.join(args.result_dir, timestamp, "saved_model")
    vocab_size = text_transform_util.DataHelper().vocab_size
    model = training(train_dataset, valid_dataset, vocab_size + 1, args.epochs, model_saved_path, log_path)

    print("\nTesting")
    from utils.make_bench_mark_data import chat_num_ls

    file_base = "total_chat_num"
    chat_num_ls = chat_num_ls
    test_benchmark_and_all(model_saved_path, args.test_start_date, args.test_end_date, args.test_tf_record_folder_name,
                           file_base, chat_num_ls)
