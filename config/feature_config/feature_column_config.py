#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python2.7
@author: zhangmeng
@file: feature_column_config.py
@time: 2019/11/11
"""
from itertools import chain
import tensorflow as tf
import numpy as np
from tensorflow import feature_column as fc
from config.feature_config.feature_infos import FEATURE_NAMES
from config.global_config import VOCABULARY_DIR
import os
from utils.date_util import DateUtil
import codecs

start_date = "20190725"
end_date = "20190813"
vocab_file_dir = os.path.join(VOCABULARY_DIR, "%s_vocabulary")

SPARSE_EMBEDDING_SIZE = 8
BUCKTE_EMBEDDING_SIZE = 8

# 配置连续变量的归一化函数和分桶边界
normalizer_dict = {
    "opp_create_obs_interval_minutes": lambda x: tf.math.log1p(tf.cast(x, tf.float32)),
    "call_record_tot_length": lambda x: tf.math.log1p(tf.cast(x, tf.float32)),
    "call_record_eff_num": None,
    "free_learn_video_length": lambda x: tf.math.log1p(tf.cast(x, tf.float32)),
    "free_learn_live_length": lambda x: tf.math.log1p(tf.cast(x, tf.float32)),
    "tot_chat_count": lambda x: tf.math.log1p(tf.cast(x, tf.float32)),
    "con_chat_count": lambda x: tf.math.log1p(tf.cast(x, tf.float32)),
    "stu_chat_count": lambda x: tf.math.log1p(tf.cast(x, tf.float32)),
    "stu_con_ratio": None,

}
bound_dict = {
    "opp_create_obs_interval_minutes": [0.0001] + list(np.arange(1, 8)) + list(np.arange(80, 120, 5) / 10),
    "call_record_tot_length": [0.0001] + list(np.arange(5, 60, 5) / 10) + list(np.arange(6, 10)),
    "call_record_eff_num": [0.0001, 3, 6, 10, 20],
    "free_learn_video_length": [0.0001] + list(np.arange(5, 60, 5) / 10) + list(np.arange(6, 12)),
    "free_learn_live_length": [0.0001] + list(np.arange(5, 60, 5) / 10) + list(np.arange(6, 12)),
    "tot_chat_count": [0.0001] + list(np.arange(5, 30, 5) / 10) + list(np.arange(3, 7)),
    "con_chat_count": [0.0001] + list(np.arange(5, 30, 5) / 10) + list(np.arange(3, 7)),
    "stu_chat_count": [0.0001] + list(np.arange(5, 30, 5) / 10) + list(np.arange(3, 7)),
    "stu_con_ratio": [0.0001] + list(np.arange(1, 10) / 100) + list(np.arange(1, 6) / 10)
}


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


# ===========================离散特征===============================
# sparse feature
# province_id = fc.categorical_column_with_hash_bucket("province_id", 100)
# city_id = fc.categorical_column_with_hash_bucket("city_id", 500)
# site_id = fc.categorical_column_with_hash_bucket("site_id", 30000)
# legion_id = fc.categorical_column_with_hash_bucket("legion_id", 1000)
# quantum_id = fc.categorical_column_with_hash_bucket("quantum_id", 1000)
# site_source = fc.categorical_column_with_hash_bucket("site_source", 5000)
# first_proj_id = fc.categorical_column_with_hash_bucket("first_proj_id", 5000)
# oppor_source = fc.categorical_column_with_hash_bucket("oppor_source", 10000)
# config_value = fc.categorical_column_with_hash_bucket("config_value", 100)
province_id = fc.categorical_column_with_vocabulary_list("province_id",
                                                         get_vocabulary_list(vocab_file_dir % "province_id",
                                                                             "province_id", start_date, end_date))
city_id = fc.categorical_column_with_vocabulary_list("city_id",
                                                     get_vocabulary_list(vocab_file_dir % "city_id", "city_id",
                                                                         start_date, end_date))
site_id = fc.categorical_column_with_vocabulary_list("site_id",
                                                     get_vocabulary_list(vocab_file_dir % "site_id", "site_id",
                                                                         start_date, end_date))
legion_id = fc.categorical_column_with_vocabulary_list("legion_id",
                                                       get_vocabulary_list(vocab_file_dir % "legion_id", "legion_id",
                                                                           start_date, end_date))
quantum_id = fc.categorical_column_with_vocabulary_list("quantum_id",
                                                        get_vocabulary_list(vocab_file_dir % "quantum_id", "quantum_id",
                                                                            start_date, end_date))
site_source = fc.categorical_column_with_vocabulary_list("site_source",
                                                         get_vocabulary_list(vocab_file_dir % "site_source",
                                                                             "site_source", start_date, end_date))
first_proj_id = fc.categorical_column_with_vocabulary_list("first_proj_id",
                                                           get_vocabulary_list(vocab_file_dir % "first_proj_id",
                                                                               "first_proj_id", start_date, end_date))
oppor_source = fc.categorical_column_with_vocabulary_list("oppor_source",
                                                          get_vocabulary_list(vocab_file_dir % "oppor_source",
                                                                              "oppor_source", start_date, end_date))
config_value = fc.categorical_column_with_vocabulary_list("config_value",
                                                          get_vocabulary_list(vocab_file_dir % "config_value",
                                                                              "config_value", start_date, end_date))

# sparse feature one_hot
province_id_one_hot = fc.indicator_column(province_id)
city_id_one_hot = fc.indicator_column(city_id)
site_id_one_hot = fc.indicator_column(site_id)
legion_id_one_hot = fc.indicator_column(legion_id)
quantum_id_one_hot = fc.indicator_column(quantum_id)
site_source_one_hot = fc.indicator_column(site_source)
first_proj_id_one_hot = fc.indicator_column(first_proj_id)
oppor_source_one_hot = fc.indicator_column(oppor_source)
config_value_one_hot = fc.indicator_column(config_value)

# sparse feature embedding
province_id_emb = fc.embedding_column(province_id, SPARSE_EMBEDDING_SIZE)
city_id_emb = fc.embedding_column(city_id, SPARSE_EMBEDDING_SIZE)
site_id_emb = fc.embedding_column(site_id, SPARSE_EMBEDDING_SIZE)
legion_id_emb = fc.embedding_column(legion_id, SPARSE_EMBEDDING_SIZE)
quantum_id_emb = fc.embedding_column(quantum_id, SPARSE_EMBEDDING_SIZE)
site_source_emb = fc.embedding_column(site_source, SPARSE_EMBEDDING_SIZE)
first_proj_id_emb = fc.embedding_column(first_proj_id, SPARSE_EMBEDDING_SIZE)
oppor_source_emb = fc.embedding_column(oppor_source, SPARSE_EMBEDDING_SIZE)
config_value_emb = fc.embedding_column(config_value, SPARSE_EMBEDDING_SIZE)

# =============================连续特征=======================================
# dense feature

# 机会注册时间间隔
opp_create_obs_interval_minutes = fc.numeric_column("opp_create_obs_interval_minutes",
                                                    normalizer_fn=normalizer_dict["opp_create_obs_interval_minutes"])
# 通话时长特征
call_record_tot_length = [fc.numeric_column(x, normalizer_fn=normalizer_dict["call_record_tot_length"]) for x in
                          FEATURE_NAMES if x.endswith("tot_length")]
# 通话有效数
call_record_eff_num = [fc.numeric_column(x, normalizer_fn=normalizer_dict["call_record_eff_num"]) for x in FEATURE_NAMES
                       if x.endswith("eff_num")]
# 免费学 - 视频时长
free_learn_video_length = [fc.numeric_column(x, normalizer_fn=normalizer_dict["free_learn_video_length"]) for x in
                           FEATURE_NAMES if x.startswith("free_learn_video_length")]
# 免费学 - 直播时长
free_learn_live_length = [fc.numeric_column(x, normalizer_fn=normalizer_dict["free_learn_live_length"]) for x in
                          FEATURE_NAMES if x.startswith("free_learn_live_length")]
# 微信统计特征
tot_chat_count = [fc.numeric_column(x, normalizer_fn=normalizer_dict["tot_chat_count"]) for x in FEATURE_NAMES if
                  x.startswith("tot_chat_count")]
con_chat_count = [fc.numeric_column(x, normalizer_fn=normalizer_dict["con_chat_count"]) for x in FEATURE_NAMES if
                  x.startswith("con_chat_count")]
stu_chat_count = [fc.numeric_column(x, normalizer_fn=normalizer_dict["stu_chat_count"]) for x in FEATURE_NAMES if
                  x.startswith("stu_chat_count")]
stu_con_ratio = [fc.numeric_column(x, normalizer_fn=normalizer_dict["stu_con_ratio"]) for x in FEATURE_NAMES if
                 x.startswith("stu_con_ratio")]

# dense embedding

# 间隔时间
opp_create_obs_interval_minutes_emb = fc.embedding_column(
    fc.bucketized_column(opp_create_obs_interval_minutes, bound_dict["opp_create_obs_interval_minutes"]),
    BUCKTE_EMBEDDING_SIZE)
# 通话时长特征
call_record_tot_length_emb = [fc.embedding_column(
    fc.bucketized_column(x, bound_dict["call_record_tot_length"]),
    BUCKTE_EMBEDDING_SIZE) for x in call_record_tot_length]
# 通话有效数
call_record_eff_num_emb = [fc.embedding_column(
    fc.bucketized_column(x, bound_dict["call_record_eff_num"]),
    BUCKTE_EMBEDDING_SIZE) for x in call_record_eff_num]
# 免费学 - 视频时长
free_learn_video_length_emb = [fc.embedding_column(
    fc.bucketized_column(x, bound_dict["free_learn_video_length"]),
    BUCKTE_EMBEDDING_SIZE) for x in free_learn_video_length]
# 免费学 - 直播时长
free_learn_live_length_emb = [fc.embedding_column(
    fc.bucketized_column(x, bound_dict["free_learn_live_length"]),
    BUCKTE_EMBEDDING_SIZE) for x in free_learn_live_length]
# 微信统计特征
tot_chat_count_emb = [fc.embedding_column(fc.bucketized_column(
    x, bound_dict["tot_chat_count"]),
    BUCKTE_EMBEDDING_SIZE) for x in tot_chat_count]
con_chat_count_emb = [fc.embedding_column(
    fc.bucketized_column(x, bound_dict["con_chat_count"]),
    BUCKTE_EMBEDDING_SIZE) for x in con_chat_count]
stu_chat_count_emb = [fc.embedding_column(
    fc.bucketized_column(x, bound_dict["stu_chat_count"]),
    BUCKTE_EMBEDDING_SIZE) for x in stu_chat_count]
stu_con_ratio_emb = [fc.embedding_column(
    fc.bucketized_column(x, bound_dict["stu_con_ratio"]),
    BUCKTE_EMBEDDING_SIZE) for x in stu_con_ratio]

# ========================= LR feature column =============================

LinnerColumns = [
    # sparse ont hot
    [
        province_id_one_hot,
        city_id_one_hot,
        site_id_one_hot,
        legion_id_one_hot,
        quantum_id_one_hot,
        site_source_one_hot,
        first_proj_id_one_hot,
        oppor_source_one_hot,
        config_value_one_hot,
        # dense log
        opp_create_obs_interval_minutes,
    ],
    call_record_tot_length,
    call_record_eff_num,
    free_learn_live_length,
    free_learn_video_length,
    tot_chat_count,
    con_chat_count,
    stu_chat_count,
    stu_con_ratio,
]
LinnerColumns = list(chain(*LinnerColumns))

# ========================= FM feature column =============================
InteractionColumns = [
    # sparse ont hot
    [
        province_id_emb,
        city_id_emb,
        site_id_emb,
        legion_id_emb,
        quantum_id_emb,
        site_source_emb,
        first_proj_id_emb,
        oppor_source_emb,
        config_value_emb,
        # dense log
        opp_create_obs_interval_minutes_emb,
    ],
    call_record_tot_length_emb,
    call_record_eff_num_emb,
    free_learn_live_length_emb,
    free_learn_video_length_emb,
    tot_chat_count_emb,
    con_chat_count_emb,
    stu_chat_count_emb,
    stu_con_ratio_emb,

]
InteractionColumns_pool = InteractionColumns
InteractionColumns = list(chain(*InteractionColumns))

# ========================= DNN feature column =============================
DNNColumns = InteractionColumns
DNNColumns_pool = InteractionColumns_pool
# 注意池化方法
# print(LinnerColumns)
# print(len(LinnerColumns))
# print(InteractionColumns)
# print(len(InteractionColumns))
