#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import jieba
from collections import Counter
import math
import re

def remove_punc(line):
    if(line.strip()==''):
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line


if __name__ == '__main__':
    fileParentPath = os.path.abspath('./..//data/rmrb/example/1946年05月/')

    print(fileParentPath)
    files = os.listdir(fileParentPath)
    print(files)
    allNewsString = ""
    for file in files:
        fileFullPath = fileParentPath + file
        try:
            with open(fileFullPath,encoding='utf8') as f:
                str = ""
                for line in iter(f):
                    str = str + line
        except:
            with open(fileFullPath, encoding='gbk') as f:
                iter_f = iter(f)
                str = ""
                for line in iter_f:
                    str = str + line
        allNewsString = allNewsString + str
    re_allNews = remove_punc(allNewsString)
    print(allNewsString)
    stopwordList = ["的", "在", "是", "之", "与", "了"]
    # 分词
    word_cut_list = " ".join([word for word in list(jieba.cut(re_allNews)) if word not in stopwordList])
    new_word_cut_list = word_cut_list.split(" ")
    # 统计每个词的频率
    word_freq = Counter(new_word_cut_list)
    print(word_freq)
    # 计算信息熵
    all_word_num = len(new_word_cut_list)
    information_entropy = 0
    for i in word_freq:
        word_freq[i] /= all_word_num
        freq = word_freq[i]
        information_entropy += freq * math.log(freq, 2)
    information_entropy = -information_entropy
    print("计算得出的信息熵为：", information_entropy)
