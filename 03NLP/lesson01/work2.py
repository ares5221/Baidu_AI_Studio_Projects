#!/usr/bin/env python
# _*_ coding:utf-8 _*_
'''
作业1-2：
（1）思考一下，假设输入一个词表里面含有N个词，输入一个长度为M的句子，那么最大前向匹配的计算复杂度是多少？
最坏情况，每个字都再词表中，那么每个都需要比较，从后向前，依次和词表里N个词进行比较，那么应该是N*(m(m+1)/2)
复杂度为o(N*M^2)
'''
# （2）给定一个句子，如何计算里面有多少种分词候选，你能给出代码实现吗？
# 最多的分词候选，那么基于最大前向匹配算法，从后到前，每匹配上1词，算1个。
input_str_len = 4
def cut_num(input_str_len):
    if(input_str_len == 1):
        return 1
    elif(input_str_len == 2):
        return 2
    else:
        return 2 * cut_num(input_str_len-1)
print(str(cut_num(input_str_len)))



import jieba
test_string = "南京市长江大桥"
word_dict= list(jieba.cut(test_string,cut_all=True))
print("句子中的单词包括：",word_dict)

def count_seg_ways(candi, remained, dict):
    if len(remained) == 0:
        print("/".join(candi))
        return 1
    count = 0
    for i in range(1, len(remained)+1):
        if remained[:i] not in dict:
            continue
        count += count_seg_ways(candi+[remained[:i]], remained[i:], dict)
    return count
print("候选数", count_seg_ways([], test_string, word_dict))
'''
（3）除了最大前向匹配和N-gram算法，你还知道其他分词算法吗，请给出一段小描述。
基于词表的分词方法
    正向最大匹配法(forward maximum matching method, FMM)
    逆向最大匹配法(backward maximum matching method, BMM)
    N-最短路径方法
基于统计模型的分词方法
基于N-gram语言模型的分词方法
基于序列标注的分词方法
基于HMM的分词方法
基于CRF的分词方法
基于词感知机的分词方法
基于深度学习的端到端的分词方法
'''