#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: text_transform_util.py
@time: 2019/10/08
"""

import re
from tensorflow import keras
import numpy as np
import json
import codecs
from config.global_config_text import *

wechat_record_feature = os.path.join(FEATURE_PATH, "wechat_record_feature")


class DataHelper(object):
    def __init__(self, vocab_path=VOCAB_PATH):
        if vocab_path and os.path.isfile(vocab_path):
            self.vocab_dict = json.load(codecs.open(vocab_path, "r", encoding="utf-8"))
            self.vocab_size = len(self.vocab_dict)
            print("词典数：", self.vocab_size)
        else:
            print("[Warning] %s 不存在，正在重新加载..." % vocab_path)
            self.prepare_vocab_dict()

    def transform_single_text_2_vector(self, text, senquence_max_len):
        # 防止text为nan
        text = text if isinstance(text, str) else ""
        x = [self.vocab_dict[each_word] if each_word in self.vocab_dict else 1 for each_word in text]
        return np.array(x[-senquence_max_len:], dtype=np.int64)

    def prepare_vocab_dict(self, raw_data_path=wechat_record_feature, vocab_file=VOCAB_PATH):
        text_preprocesser = keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
        filenames = [os.path.join(raw_data_path, x) for x in os.listdir(raw_data_path)]
        filenames = sorted(filenames)[::5]
        if len(filenames) > 20:
            filenames = sorted(np.random.choice(filenames, 20))
        for data_file in filenames:
            print(data_file)
            x_text = []
            with codecs.open(data_file, "r", "utf-8") as fin:
                for line in fin:
                    arr = line.strip().split("\t")
                    if len(arr) < 2:
                        continue
                    x_text.append(' '.join(string for string in arr[-1]))
            text_preprocesser.fit_on_texts(x_text)
        word_dict = text_preprocesser.word_index
        # 添加空格到字典中，用于断句
        word_dict[" "] = len(word_dict) + 1
        json.dump(word_dict, open(vocab_file, 'w', encoding="utf-8"))
        word_dict = json.load(open(VOCAB_PATH, 'r', encoding='utf-8'))
        self.vocab_dict = word_dict
        self.vocab_size = len(self.vocab_dict)
        print("vocab dumps finished! word num:", self.vocab_size)

    def text_preprocess(self, text):
        """
        Clean and segment the text.
        Return a new text.
        """
        text = re.sub(r"[\d+\s+\.!\/_,?=\$%\^\)*\(\+\"\'\+——！:；，。？、~@#%……&*（）·¥\-\|\\《》〈〉～]",
                      "", text)
        text = re.sub("[<>]", "", text)
        text = re.sub("[a-zA-Z0-9]", "", text)
        text = re.sub(r"\s", "", text)
        if not text:
            return ''
        return ' '.join(string for string in text)


if __name__ == '__main__':
    helper = DataHelper()
    word_dict = helper.vocab_dict

    print(word_dict)
    print("空格是否在字典中用于断句：", " " in word_dict)
