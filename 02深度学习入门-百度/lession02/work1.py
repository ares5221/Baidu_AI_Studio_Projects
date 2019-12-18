#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#设置图片大小
plt.figure()
# x是1维数组，数组大小是从-10. 到10.的实数，每隔0.1取一个点
x = np.arange(-10, 10, 0.1)
# 计算 tanh函数
s = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(- x))

#########################################################
# 以下部分为画图程序
# 画出函数曲线
plt.plot(x, s, color='r')
# 添加文字说明
plt.text(-5., 0.9, r'$y=\tanh(x)$', fontsize=13)
# 设置坐标轴格式
currentAxis=plt.gca()
currentAxis.xaxis.set_label_text('x', fontsize=15)
currentAxis.yaxis.set_label_text('y', fontsize=15)
plt.show()