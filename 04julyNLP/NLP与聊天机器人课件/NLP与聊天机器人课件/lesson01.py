#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import nltk
# 下载语料
# nltk.download('punkt')

from nltk.corpus import brown
print(brown.categories())
import nltk
sentence = "hello, world"
tokens = nltk.word_tokenize(sentence)
print(tokens)
