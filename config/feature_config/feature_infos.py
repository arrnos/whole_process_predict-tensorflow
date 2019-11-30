# -*- coding: utf-8 -*-

"""
@author: liutao
@file: feature_infos.py
@time: 2019/10/10 17:17
"""

import numpy as np
from tensorflow import feature_column as fc
import tensorflow as tf

NUMERIC = "numeric"
CATEGORICAL = "categorical"

FEATURE_INFOS = {

    0: ["label", "numeric", np.int64, 0, 0],

    # 特征数据路径
    # /home/yanxin/project_data/whole_process_extend_sample/feature_file/merged_feature

    # 类别型特征
    1: ["province_id", "categorical", str, ""],
    2: ["city_id", "categorical", str, ""],
    3: ["site_id", "categorical", str, ""],
    4: ["legion_id", "categorical", str, ""],
    5: ["quantum_id", "categorical", str, ""],
    6: ["site_source", "categorical", str, ""],
    7: ["first_proj_id", "categorical", str, ""],
    8: ["oppor_source", "categorical", str, ""],
    9: ["config_value", "categorical", str, ""],
    # 咨询师静态特征待添加

    # 数值型特征
    # 机会注册时间间隔特征
    10: ["opp_create_obs_interval_minutes", "numeric", np.float32, 0.0],

    # 通话时长特征
    11: ["call_record_0_eff_num", "numeric", np.int64, 0],
    12: ["call_record_0_tot_length", "numeric", np.int64, 0],
    13: ["call_record_1_eff_num", "numeric", np.int64, 0],
    14: ["call_record_1_tot_length", "numeric", np.int64, 0],
    15: ["call_record_2_eff_num", "numeric", np.int64, 0],
    16: ["call_record_2_tot_length", "numeric", np.int64, 0],
    17: ["call_record_3_eff_num", "numeric", np.int64, 0],
    18: ["call_record_3_tot_length", "numeric", np.int64, 0],
    19: ["call_record_4_eff_num", "numeric", np.int64, 0],
    20: ["call_record_4_tot_length", "numeric", np.int64, 0],
    21: ["call_record_5_eff_num", "numeric", np.int64, 0],
    22: ["call_record_5_tot_length", "numeric", np.int64, 0],
    23: ["call_record_6_eff_num", "numeric", np.int64, 0],
    24: ["call_record_6_tot_length", "numeric", np.int64, 0],
    25: ["call_record_7_eff_num", "numeric", np.int64, 0],
    26: ["call_record_7_tot_length", "numeric", np.int64, 0],
    27: ["call_record_8_eff_num", "numeric", np.int64, 0],
    28: ["call_record_8_tot_length", "numeric", np.int64, 0],
    29: ["call_record_9_eff_num", "numeric", np.int64, 0],
    30: ["call_record_9_tot_length", "numeric", np.int64, 0],
    31: ["call_record_10_eff_num", "numeric", np.int64, 0],
    32: ["call_record_10_tot_length", "numeric", np.int64, 0],
    33: ["call_record_11_eff_num", "numeric", np.int64, 0],
    34: ["call_record_11_tot_length", "numeric", np.int64, 0],
    35: ["call_record_12_eff_num", "numeric", np.int64, 0],
    36: ["call_record_12_tot_length", "numeric", np.int64, 0],
    37: ["call_record_13_eff_num", "numeric", np.int64, 0],
    38: ["call_record_13_tot_length", "numeric", np.int64, 0],

    # 免费学打点特征
    39: ["free_learn_video_length_0", "numeric", np.int64, 0],
    40: ["free_learn_live_length_0", "numeric", np.int64, 0],
    41: ["free_learn_total_length_0", "numeric", np.int64, 0],
    42: ["free_learn_video_length_1", "numeric", np.int64, 0],
    43: ["free_learn_live_length_1", "numeric", np.int64, 0],
    44: ["free_learn_total_length_1", "numeric", np.int64, 0],
    45: ["free_learn_video_length_2", "numeric", np.int64, 0],
    46: ["free_learn_live_length_2", "numeric", np.int64, 0],
    47: ["free_learn_total_length_2", "numeric", np.int64, 0],
    48: ["free_learn_video_length_3", "numeric", np.int64, 0],
    49: ["free_learn_live_length_3", "numeric", np.int64, 0],
    50: ["free_learn_total_length_3", "numeric", np.int64, 0],
    51: ["free_learn_video_length_4", "numeric", np.int64, 0],
    52: ["free_learn_live_length_4", "numeric", np.int64, 0],
    53: ["free_learn_total_length_4", "numeric", np.int64, 0],
    54: ["free_learn_video_length_5", "numeric", np.int64, 0],
    55: ["free_learn_live_length_5", "numeric", np.int64, 0],
    56: ["free_learn_total_length_5", "numeric", np.int64, 0],
    57: ["free_learn_video_length_6", "numeric", np.int64, 0],
    58: ["free_learn_live_length_6", "numeric", np.int64, 0],
    59: ["free_learn_total_length_6", "numeric", np.int64, 0],
    60: ["free_learn_video_length_7", "numeric", np.int64, 0],
    61: ["free_learn_live_length_7", "numeric", np.int64, 0],
    62: ["free_learn_total_length_7", "numeric", np.int64, 0],
    63: ["free_learn_video_length_8", "numeric", np.int64, 0],
    64: ["free_learn_live_length_8", "numeric", np.int64, 0],
    65: ["free_learn_total_length_8", "numeric", np.int64, 0],
    66: ["free_learn_video_length_9", "numeric", np.int64, 0],
    67: ["free_learn_live_length_9", "numeric", np.int64, 0],
    68: ["free_learn_total_length_9", "numeric", np.int64, 0],
    69: ["free_learn_video_length_10", "numeric", np.int64, 0],
    70: ["free_learn_live_length_10", "numeric", np.int64, 0],
    71: ["free_learn_total_length_10", "numeric", np.int64, 0],
    72: ["free_learn_video_length_11", "numeric", np.int64, 0],
    73: ["free_learn_live_length_11", "numeric", np.int64, 0],
    74: ["free_learn_total_length_11", "numeric", np.int64, 0],
    75: ["free_learn_video_length_12", "numeric", np.int64, 0],
    76: ["free_learn_live_length_12", "numeric", np.int64, 0],
    77: ["free_learn_total_length_12", "numeric", np.int64, 0],
    78: ["free_learn_video_length_13", "numeric", np.int64, 0],
    79: ["free_learn_live_length_13", "numeric", np.int64, 0],
    80: ["free_learn_total_length_13", "numeric", np.int64, 0],

    # 微信统计特征
    81: ["tot_chat_count_0", "numeric", np.int64, 0],
    82: ["con_chat_count_0", "numeric", np.int64, 0],
    83: ["stu_chat_count_0", "numeric", np.int64, 0],
    84: ["stu_con_ratio_0", "numeric", np.float32, 0.0],
    85: ["tot_chat_count_1", "numeric", np.int64, 0],
    86: ["con_chat_count_1", "numeric", np.int64, 0],
    87: ["stu_chat_count_1", "numeric", np.int64, 0],
    88: ["stu_con_ratio_1", "numeric", np.float32, 0.0],
    89: ["tot_chat_count_2", "numeric", np.int64, 0],
    90: ["con_chat_count_2", "numeric", np.int64, 0],
    91: ["stu_chat_count_2", "numeric", np.int64, 0],
    92: ["stu_con_ratio_2", "numeric", np.float32, 0.0],
    93: ["tot_chat_count_3", "numeric", np.int64, 0],
    94: ["con_chat_count_3", "numeric", np.int64, 0],
    95: ["stu_chat_count_3", "numeric", np.int64, 0],
    96: ["stu_con_ratio_3", "numeric", np.float32, 0.0],
    97: ["tot_chat_count_4", "numeric", np.int64, 0],
    98: ["con_chat_count_4", "numeric", np.int64, 0],
    99: ["stu_chat_count_4", "numeric", np.int64, 0],
    100: ["stu_con_ratio_4", "numeric", np.float32, 0.0],
    101: ["tot_chat_count_5", "numeric", np.int64, 0],
    102: ["con_chat_count_5", "numeric", np.int64, 0],
    103: ["stu_chat_count_5", "numeric", np.int64, 0],
    104: ["stu_con_ratio_5", "numeric", np.float32, 0.0],
    105: ["tot_chat_count_6", "numeric", np.int64, 0],
    106: ["con_chat_count_6", "numeric", np.int64, 0],
    107: ["stu_chat_count_6", "numeric", np.int64, 0],
    108: ["stu_con_ratio_6", "numeric", np.float32, 0.0],
    109: ["tot_chat_count_7", "numeric", np.int64, 0],
    110: ["con_chat_count_7", "numeric", np.int64, 0],
    111: ["stu_chat_count_7", "numeric", np.int64, 0],
    112: ["stu_con_ratio_7", "numeric", np.float32, 0.0],
    113: ["tot_chat_count_8", "numeric", np.int64, 0],
    114: ["con_chat_count_8", "numeric", np.int64, 0],
    115: ["stu_chat_count_8", "numeric", np.int64, 0],
    116: ["stu_con_ratio_8", "numeric", np.float32, 0.0],
    117: ["tot_chat_count_9", "numeric", np.int64, 0],
    118: ["con_chat_count_9", "numeric", np.int64, 0],
    119: ["stu_chat_count_9", "numeric", np.int64, 0],
    120: ["stu_con_ratio_9", "numeric", np.float32, 0.0],
    121: ["tot_chat_count_10", "numeric", np.int64, 0],
    122: ["con_chat_count_10", "numeric", np.int64, 0],
    123: ["stu_chat_count_10", "numeric", np.int64, 0],
    124: ["stu_con_ratio_10", "numeric", np.float32, 0.0],
    125: ["tot_chat_count_11", "numeric", np.int64, 0],
    126: ["con_chat_count_11", "numeric", np.int64, 0],
    127: ["stu_chat_count_11", "numeric", np.int64, 0],
    128: ["stu_con_ratio_11", "numeric", np.float32, 0.0],
    129: ["tot_chat_count_12", "numeric", np.int64, 0],
    130: ["con_chat_count_12", "numeric", np.int64, 0],
    131: ["stu_chat_count_12", "numeric", np.int64, 0],
    132: ["stu_con_ratio_12", "numeric", np.float32, 0.0],
    133: ["tot_chat_count_13", "numeric", np.int64, 0],
    134: ["con_chat_count_13", "numeric", np.int64, 0],
    135: ["stu_chat_count_13", "numeric", np.int64, 0],
    136: ["stu_con_ratio_13", "numeric", np.float32, 0.0],

    # 文本特征
    137: ["audio", "categorical", str, ""],
    138: ["wechat", "categorical", str, ""],

    # 极速APP行为特征
    # info_session特征
    # 大熊和表单统计特征

}


def get_dict_feature_index_and_type():
    feature_name_list = []
    feature_type_list = []
    feature_dtype_list = []
    feature_default_list = []
    feature_input_dict = {}

    for index, tmp_ls in sorted(FEATURE_INFOS.items(), key=lambda x: x[0]):
        feature_name = tmp_ls[0]
        feature_type = tmp_ls[1]
        feature_dtype = tmp_ls[2]
        default_value = tmp_ls[3]
        feature_name_list.append(feature_name)
        feature_type_list.append(feature_type)
        feature_dtype_list.append(feature_dtype)
        feature_default_list.append(default_value)
        if feature_name == "label":
            continue
        elif feature_dtype == str:
            feature_input_dict[feature_name] = tf.keras.Input(name=feature_name, shape=(1,), dtype=tf.string)
        elif feature_dtype == np.int64:
            feature_input_dict[feature_name] = tf.keras.Input(name=feature_name, shape=(1,), dtype=tf.int64)
        elif feature_dtype == np.float32:
            feature_input_dict[feature_name] = tf.keras.Input(name=feature_name, shape=(1,), dtype=tf.float32)
        else:
            raise ValueError("expect data type in (float32,float32,str),but given %s" % feature_dtype)

    return feature_name_list, feature_type_list, feature_dtype_list, feature_default_list, feature_input_dict


FEATURE_NAMES, FEATURE_TYPES, FEATURE_DTYPES, FEATURE_DEFAULTS, FEATURE_INPUT_DICT = get_dict_feature_index_and_type()

print(FEATURE_INPUT_DICT)
