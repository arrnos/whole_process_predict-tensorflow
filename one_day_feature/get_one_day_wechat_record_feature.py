# -*- coding:UTF-8 -*-
"""

"""
import codecs
import json

from config.global_config_text import *
from utils.date_util import DateUtil

HISTORY_WECHAT_RECORD_DELTA_DAY = 7

merged_wechat_record_data_path = MERGED_RAW_WECHAT_RECORD_PATH
merged_raw_data_path = MERGED_RAW_DATA_PATH
wechat_record_feature_path = os.path.join(PROJECT_DATA_DIR, "feature_file", "wechat_record_feature")

tmp_merged_wechat_record_data_file = os.path.join(merged_wechat_record_data_path, "merged_wechat_data_%s")
tmp_merged_raw_data_file = os.path.join(merged_raw_data_path, "merged_raw_data_%s")
tmp_wechat_record_feature_file = os.path.join(wechat_record_feature_path, "wechat_record_feature_%s")


def merge_wechat_records_by_opp_id(start_date, end_date):
    print("merge_wechat_records_by_opp_id..")
    date_ls = DateUtil.get_every_date(start_date, end_date)
    rs_dict = {}
    for date in date_ls:
        print(date)
        wechat_file = tmp_merged_wechat_record_data_file % date
        with codecs.open(wechat_file, 'r', 'utf-8') as fin:
            for line in fin:
                arr = line.strip().split("\t")
                opp_id = arr[0]
                chat_ls = []
                for chat_str in arr[1:]:
                    try:
                        chat_dict = json.loads(chat_str, encoding='utf-8')
                    except:
                        # print("wechat record dict can't parse by json:", chat_str)
                        continue
                    chat_ls.append(chat_dict)
                if opp_id not in rs_dict:
                    rs_dict[opp_id] = chat_ls
                else:
                    rs_dict[opp_id].extend(chat_ls)

    print("sort chat by create_time..")
    for opp_id, chat_ls in rs_dict.items():
        chat_ls.sort(key=lambda json_dict: json_dict["create_time"])
    return rs_dict


def get_one_day_wechat_record_feature(date):
    wechat_record_start_date = DateUtil.get_relative_delta_time_str(date, -HISTORY_WECHAT_RECORD_DELTA_DAY)
    wechat_record_dict = merge_wechat_records_by_opp_id(wechat_record_start_date, date)

    merge_raw_data_file = tmp_merged_raw_data_file % date
    wechat_record_featue_file = tmp_wechat_record_feature_file % date
    with codecs.open(merge_raw_data_file, 'r', 'utf-8') as fin, codecs.open(wechat_record_featue_file, "w",
                                                                            'utf-8') as fout:
        for line in fin:
            arr = line.strip().split("\t")
            label = arr[0]
            time_observe_point = arr[1]
            # time_observe_point = datetime.strptime(arr[1],"%Y-%m-%d %H:%M:%S")
            opp_id = arr[6]
            if opp_id not in wechat_record_dict:
                result = "\t".join(map(str, [label, opp_id, time_observe_point] + [0] * 3 + [""]))
                fout.write(result + "\n")
                continue
            wechat_stat_ls = wechat_record_precess(wechat_record_dict[opp_id], time_observe_point)
            result = "\t".join(map(str, [label, opp_id, time_observe_point] + wechat_stat_ls))
            fout.write(result + "\n")


# 微信聊天数据统计特征和聊天数据拼接
def wechat_record_precess(wechat_record_ls, time_observe_point):
    # 单句聊天文本处理逻辑
    def cleared_chat_record(chat_str):
        chinese_sentence = "".join([word for word in chat_str if u'\u4e00' <= word <= u'\u9fff'])
        # 超长语句截断
        if len(chinese_sentence) > 70:
            chinese_sentence = chinese_sentence[:20] + chinese_sentence[-20:]

        return chinese_sentence

    chat_str_ls = []
    student_chat_num = 0
    teacher_chat_num = 0
    for tmp_chat_dict in wechat_record_ls:
        send_type = tmp_chat_dict["send_type"]
        create_time = tmp_chat_dict["create_time"]
        chat_record = tmp_chat_dict["chat_record"]
        if create_time > time_observe_point:
            continue
        if send_type == "0":
            teacher_chat_num += 1
        if send_type == "1":
            student_chat_num += 1
        cleared_chat = cleared_chat_record(chat_record)
        chat_str_ls.append(cleared_chat)
    return [student_chat_num, teacher_chat_num, student_chat_num + teacher_chat_num, " ".join(chat_str_ls)]


def get_multi_day_wechat_record_feature(start_date, end_date):
    for date in DateUtil.get_every_date(start_date, end_date):
        print(date)
        get_one_day_wechat_record_feature(date)


def main():
    import sys
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    get_multi_day_wechat_record_feature(start_date, end_date)


if __name__ == "__main__":
    main()
