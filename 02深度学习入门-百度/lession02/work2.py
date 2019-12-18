#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np

p = np.random.randn(10, 10)
# print(p)
q = p >0
print('q的数据类型：',type(q), type(q[0,0]))
print('q的元素的取值', q)
pp = p.reshape(100)
positive_num = 0
for idx in range(pp.size):
    if pp[idx] >0:
        positive_num +=1
print('随机数构成的矩阵中元素大于0的个数',positive_num)
