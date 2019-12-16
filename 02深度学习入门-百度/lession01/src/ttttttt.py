#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt


def load_data():
    # 从文件导入数据
    datafile = './../data/housing.data'
    data = np.fromfile(datafile, sep=' ')
    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)
    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                               training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
    # print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w1 = np.random.randn(num_of_weights, 13)
        self.b1 = np.zeros((1, 13))
        self.w2 = np.random.randn(13, 1)
        self.b2 = 0.
    # 两层网络前向计算
    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        z2 = np.dot(z1, self.w2) + self.b2
        self.z1 = z1
        return z2
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    # 两层网络梯度值更新
    def gradient(self, x, y):
        z2 = self.forward(x)
        N = x.shape[0]
        self.gradient_w2 = 1. / N * np.dot(self.z1.T, z2 - y)
        self.gradient_b2 = 1. / N * np.sum(z2 - y)
        gradient_z1 = 1. / N * np.dot((z2 - y), self.w2.T)
        self.gradient_w1 = np.dot(x.T, gradient_z1)
        self.gradient_b1 = np.sum(gradient_z1, axis=0)
        self.gradient_b1 = self.gradient_b1[np.newaxis, :]

    # 更新两层网络参数
    def update(self, eta=0.01):
        self.w2 = self.w2 - eta * self.gradient_w2
        self.b2 = self.b2 - eta * self.gradient_b2
        self.w1 = self.w1 - eta * self.gradient_w1
        self.b1 = self.b1 - eta * self.gradient_b1


    def train(self, training_data, num_epoches, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epoches):
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱，
        # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
        # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
            # print(self.w.shape)
            # print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                self.gradient(x, y)
                self.update(eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, loss))
        return losses


# 获取数据
train_data, test_data = load_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data, num_epoches=50, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
