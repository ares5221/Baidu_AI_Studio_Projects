#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import matplotlib.pyplot as plt
import numpy as np


'''
sigmoid函数只有在接近0的地方，导数才比较大，最大为1/4，在x趋向
正负无穷的地方，导数为0

反向传播每次都乘一个小于1的数，若多层网络，会导致
较靠前的层的梯度变得非常小

梯度值衰减到接近于0的现象也就是梯度消失
'''


plt.figure()
x = np.arange(-100,100,0.1)
y = 1/(2 + np.exp(x) + np.exp(-x))
plt.plot(x, y, color='r')
# 添加文字说明
plt.text(-5., 0.9, r'$y=\tanh(x)$', fontsize=13)
# 设置坐标轴格式
currentAxis=plt.gca()
currentAxis.xaxis.set_label_text('x', fontsize=15)
currentAxis.yaxis.set_label_text('y', fontsize=15)
plt.show()
