# -*- coding: utf-8 -*-

"""
@author: liutao
@file: feature_infos.py
@time: 2019/10/10 17:17
"""

import numpy as np

NUMERIC = "numeric"
CATEGORICAL = "categorical"

FEATURE_INFOS = {

    0: ["label", "numeric", np.int64],

    # 特征数据路径
    # /home/yanxin/project_data/whole_process_extend_sample/feature_file/merged_feature

    # 类别型特征
    1: ["province_id", "categorical", str],
    2: ["city_id", "categorical", str],
    3: ["site_id", "categorical", str],
    4: ["legion_id", "categorical", str],
    5: ["quantum_id", "categorical", str],
    6: ["site_source", "categorical", str],
    7: ["first_proj_id", "categorical", str],
    8: ["oppor_source", "categorical", str],
    9: ["config_value", "categorical", str],
    # 咨询师静态特征待添加

    # 数值型特征
    # 机会注册时间间隔特征
    10: ["opp_create_obs_interval_minutes", "numeric", np.float32],

    # 通话时长特征
    11: ["call_record_0_eff_num", "numeric", np.int64],
    12: ["call_record_0_tot_length", "numeric", np.int64],
    13: ["call_record_1_eff_num", "numeric", np.int64],
    14: ["call_record_1_tot_length", "numeric", np.int64],
    15: ["call_record_2_eff_num", "numeric", np.int64],
    16: ["call_record_2_tot_length", "numeric", np.int64],
    17: ["call_record_3_eff_num", "numeric", np.int64],
    18: ["call_record_3_tot_length", "numeric", np.int64],
    19: ["call_record_4_eff_num", "numeric", np.int64],
    20: ["call_record_4_tot_length", "numeric", np.int64],
    21: ["call_record_5_eff_num", "numeric", np.int64],
    22: ["call_record_5_tot_length", "numeric", np.int64],
    23: ["call_record_6_eff_num", "numeric", np.int64],
    24: ["call_record_6_tot_length", "numeric", np.int64],
    25: ["call_record_7_eff_num", "numeric", np.int64],
    26: ["call_record_7_tot_length", "numeric", np.int64],
    27: ["call_record_8_eff_num", "numeric", np.int64],
    28: ["call_record_8_tot_length", "numeric", np.int64],
    29: ["call_record_9_eff_num", "numeric", np.int64],
    30: ["call_record_9_tot_length", "numeric", np.int64],
    31: ["call_record_10_eff_num", "numeric", np.int64],
    32: ["call_record_10_tot_length", "numeric", np.int64],
    33: ["call_record_11_eff_num", "numeric", np.int64],
    34: ["call_record_11_tot_length", "numeric", np.int64],
    35: ["call_record_12_eff_num", "numeric", np.int64],
    36: ["call_record_12_tot_length", "numeric", np.int64],
    37: ["call_record_13_eff_num", "numeric", np.int64],
    38: ["call_record_13_tot_length", "numeric", np.int64],

    # 免费学打点特征
    39: ["free_learn_video_length_0", "numeric", np.int64],
    40: ["free_learn_live_length_0", "numeric", np.int64],
    41: ["free_learn_total_length_0", "numeric", np.int64],
    42: ["free_learn_video_length_1", "numeric", np.int64],
    43: ["free_learn_live_length_1", "numeric", np.int64],
    44: ["free_learn_total_length_1", "numeric", np.int64],
    45: ["free_learn_video_length_2", "numeric", np.int64],
    46: ["free_learn_live_length_2", "numeric", np.int64],
    47: ["free_learn_total_length_2", "numeric", np.int64],
    48: ["free_learn_video_length_3", "numeric", np.int64],
    49: ["free_learn_live_length_3", "numeric", np.int64],
    50: ["free_learn_total_length_3", "numeric", np.int64],
    51: ["free_learn_video_length_4", "numeric", np.int64],
    52: ["free_learn_live_length_4", "numeric", np.int64],
    53: ["free_learn_total_length_4", "numeric", np.int64],
    54: ["free_learn_video_length_5", "numeric", np.int64],
    55: ["free_learn_live_length_5", "numeric", np.int64],
    56: ["free_learn_total_length_5", "numeric", np.int64],
    57: ["free_learn_video_length_6", "numeric", np.int64],
    58: ["free_learn_live_length_6", "numeric", np.int64],
    59: ["free_learn_total_length_6", "numeric", np.int64],
    60: ["free_learn_video_length_7", "numeric", np.int64],
    61: ["free_learn_live_length_7", "numeric", np.int64],
    62: ["free_learn_total_length_7", "numeric", np.int64],
    63: ["free_learn_video_length_8", "numeric", np.int64],
    64: ["free_learn_live_length_8", "numeric", np.int64],
    65: ["free_learn_total_length_8", "numeric", np.int64],
    66: ["free_learn_video_length_9", "numeric", np.int64],
    67: ["free_learn_live_length_9", "numeric", np.int64],
    68: ["free_learn_total_length_9", "numeric", np.int64],
    69: ["free_learn_video_length_10", "numeric", np.int64],
    70: ["free_learn_live_length_10", "numeric", np.int64],
    71: ["free_learn_total_length_10", "numeric", np.int64],
    72: ["free_learn_video_length_11", "numeric", np.int64],
    73: ["free_learn_live_length_11", "numeric", np.int64],
    74: ["free_learn_total_length_11", "numeric", np.int64],
    75: ["free_learn_video_length_12", "numeric", np.int64],
    76: ["free_learn_live_length_12", "numeric", np.int64],
    77: ["free_learn_total_length_12", "numeric", np.int64],
    78: ["free_learn_video_length_13", "numeric", np.int64],
    79: ["free_learn_live_length_13", "numeric", np.int64],
    80: ["free_learn_total_length_13", "numeric", np.int64],

    # 微信统计特征
    81: ["tot_chat_count_0", "numeric", np.int64],
    82: ["con_chat_count_0", "numeric", np.int64],
    83: ["stu_chat_count_0", "numeric", np.int64],
    84: ["stu_con_ratio_0", "numeric", np.float32],
    85: ["tot_chat_count_1", "numeric", np.int64],
    86: ["con_chat_count_1", "numeric", np.int64],
    87: ["stu_chat_count_1", "numeric", np.int64],
    88: ["stu_con_ratio_1", "numeric", np.float32],
    89: ["tot_chat_count_2", "numeric", np.int64],
    90: ["con_chat_count_2", "numeric", np.int64],
    91: ["stu_chat_count_2", "numeric", np.int64],
    92: ["stu_con_ratio_2", "numeric", np.float32],
    93: ["tot_chat_count_3", "numeric", np.int64],
    94: ["con_chat_count_3", "numeric", np.int64],
    95: ["stu_chat_count_3", "numeric", np.int64],
    96: ["stu_con_ratio_3", "numeric", np.float32],
    97: ["tot_chat_count_4", "numeric", np.int64],
    98: ["con_chat_count_4", "numeric", np.int64],
    99: ["stu_chat_count_4", "numeric", np.int64],
    100: ["stu_con_ratio_4", "numeric", np.float32],
    101: ["tot_chat_count_5", "numeric", np.int64],
    102: ["con_chat_count_5", "numeric", np.int64],
    103: ["stu_chat_count_5", "numeric", np.int64],
    104: ["stu_con_ratio_5", "numeric", np.float32],
    105: ["tot_chat_count_6", "numeric", np.int64],
    106: ["con_chat_count_6", "numeric", np.int64],
    107: ["stu_chat_count_6", "numeric", np.int64],
    108: ["stu_con_ratio_6", "numeric", np.float32],
    109: ["tot_chat_count_7", "numeric", np.int64],
    110: ["con_chat_count_7", "numeric", np.int64],
    111: ["stu_chat_count_7", "numeric", np.int64],
    112: ["stu_con_ratio_7", "numeric", np.float32],
    113: ["tot_chat_count_8", "numeric", np.int64],
    114: ["con_chat_count_8", "numeric", np.int64],
    115: ["stu_chat_count_8", "numeric", np.int64],
    116: ["stu_con_ratio_8", "numeric", np.float32],
    117: ["tot_chat_count_9", "numeric", np.int64],
    118: ["con_chat_count_9", "numeric", np.int64],
    119: ["stu_chat_count_9", "numeric", np.int64],
    120: ["stu_con_ratio_9", "numeric", np.float32],
    121: ["tot_chat_count_10", "numeric", np.int64],
    122: ["con_chat_count_10", "numeric", np.int64],
    123: ["stu_chat_count_10", "numeric", np.int64],
    124: ["stu_con_ratio_10", "numeric", np.float32],
    125: ["tot_chat_count_11", "numeric", np.int64],
    126: ["con_chat_count_11", "numeric", np.int64],
    127: ["stu_chat_count_11", "numeric", np.int64],
    128: ["stu_con_ratio_11", "numeric", np.float32],
    129: ["tot_chat_count_12", "numeric", np.int64],
    130: ["con_chat_count_12", "numeric", np.int64],
    131: ["stu_chat_count_12", "numeric", np.int64],
    132: ["stu_con_ratio_12", "numeric", np.float32],
    133: ["tot_chat_count_13", "numeric", np.int64],
    134: ["con_chat_count_13", "numeric", np.int64],
    135: ["stu_chat_count_13", "numeric", np.int64],
    136: ["stu_con_ratio_13", "numeric", np.float32],

    # 文本特征
    137: ["audio", "categorical", str],
    138: ["wechat", "categorical", str],

    # 极速APP行为特征
    # info_session特征
    # 大熊和表单统计特征

}


def get_dict_feature_index_and_type():
    dict_feature_index = dict()
    dict_feature_type = dict()
    dict_feature_dtype = dict()
    for index, tmp_ls in FEATURE_INFOS.items():
        feature_name = tmp_ls[0]
        feature_type = tmp_ls[1]
        feature_dtype = tmp_ls[2]
        dict_feature_index[feature_name] = index
        dict_feature_type[feature_name] = feature_type
        dict_feature_dtype[feature_name] = feature_dtype

    return dict_feature_index, dict_feature_type, dict_feature_dtype


DICT_FEATURE_INDEX, DICT_FEATURE_TYPE, DICT_FEATURE_DTYPE = get_dict_feature_index_and_type()
FEATURE_NAMES = list(DICT_FEATURE_INDEX.keys())
