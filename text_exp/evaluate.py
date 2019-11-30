from tensorflow import keras
import argparse
import numpy as np
from config.global_config_text import *
from text_exp.dataset_read_config import *


def evaluate(model_saved_path, valid_dataset):
    model = keras.models.load_model(model_saved_path)
    result = model.evaluate(valid_dataset)
    print("\nEvaluate result:", result)

    y_pred = model.predict(valid_dataset)
    y_true = valid_dataset.map(lambda x, y: y).unbatch()
    y_true_ls = []
    for i in y_true:
        y_true_ls.append(i.numpy())
    from sklearn.metrics import roc_auc_score
    y_true = np.array(y_true_ls)
    auc = roc_auc_score(y_true, y_pred)

    print("AUC:{},正样本数:{},总样本量:{},正样本占比:{}".format(auc, sum(y_true), len(y_true), sum(y_true) / len(y_true)))


def evaluate_model_from_path(model_saved_path, test_tf_record_path, start_date, end_date):
    assert os.path.exists(model_saved_path) and os.path.isdir(test_tf_record_path)

    valid_dataset = tfRecorder.get_dataset_from_path(test_tf_record_path, feature_ls, label_name=label_name,
                                                     start_date=start_date, end_date=end_date,
                                                     batch_size=evaluate_batch_size, padding=padding)
    print("[%s]" % os.path.basename(test_tf_record_path))
    evaluate(model_saved_path, valid_dataset)


def evaluate_model_from_files(model_saved_path, test_tf_record_files, data_info_csv_path):
    assert os.path.exists(model_saved_path) and test_tf_record_files

    valid_dataset = tfRecorder._get_dataset(test_tf_record_files, data_info_csv_path, feature_ls, label_name=label_name,
                                            batch_size=evaluate_batch_size, padding=padding)
    evaluate(model_saved_path, valid_dataset)


def test_benchmark_and_all(model_path, start_date, end_date, test_tf_record_folder_name, file_base, chat_num_ls):
    all_test_files = []
    data_info_csv_path = ""
    for tmp_num in map(str, chat_num_ls):
        test_tf_record_path = os.path.join(TF_RECORD_PATH, test_tf_record_folder_name, "%s_%s" % (file_base, tmp_num))
        assert os.path.isdir(test_tf_record_path)
        test_files = [os.path.join(test_tf_record_path,x) for x in os.listdir(test_tf_record_path) if
                      start_date <= x.replace(".tfrecord","").split("_")[-1] <= end_date]
        data_info_csv_path = os.path.join(test_tf_record_path, "data_info.csv")

        print(os.path.basename(test_tf_record_path))
        evaluate_model_from_files(model_path, test_files, data_info_csv_path)
        all_test_files.extend(test_files)
    print("[ALL Benchmark]")
    evaluate_model_from_files(model_path, all_test_files, data_info_csv_path)
    print("[ALL Testdata]")
    evaluate_model_from_path(model_path,os.path.join(TF_RECORD_PATH,"wechat_record_feature_remove_absent"),start_date,end_date)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate model..")
    parser.add_argument("-model_saved_path", "--model_saved_path", type=str, default="", help="model_saved_path")
    parser.add_argument("-start_date", "--start_date", type=str, default="20190801", help="start_date")
    parser.add_argument("-end_date", "--end_date", type=str, default="20190930", help="end_date")
    parser.add_argument("-test_on_benchmark", "--test_on_benchmark", type=int, default=0, help="test_on_benchmark")
    parser.add_argument("-test_tf_record_path", "--test_tf_record_path", type=str, default="",
                        help="test_tf_record_path")
    parser.add_argument("-test_tf_record_folder_name", "--test_tf_record_folder_name", type=str, default="",
                        help="test_tf_record_folder_name")

    args = parser.parse_args()
    print("Argument:", args, "\n")
    print(args.test_on_benchmark)

    assert os.path.exists(args.model_saved_path)
    assert args.start_date <= args.end_date and "" not in [args.start_date, args.end_date]

    file_base = "total_chat_num"
    if args.test_on_benchmark:
        test_benchmark_and_all(args.model_saved_path, args.start_date, args.end_date, args.test_tf_record_folder_name,
                               file_base, chat_num_ls)
    else:
        assert os.path.isdir(args.test_tf_record_path)
        evaluate_model_from_path(args.model_saved_path, args.test_tf_record_path, args.start_date, args.end_date)
