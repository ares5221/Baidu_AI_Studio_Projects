#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np

ss = np.arange(9)
print(ss)

dd = ss.reshape(3,3)
print(dd)
ee = np.ones([3,3])
# print(ee)
# print(dd.dot(ee))
print(np.diag(dd))
print(np.trace(dd))
print(np.linalg.det(dd))

print(np.linalg.eig(dd))

tmp = np.random.randn(3,3)
print(tmp)
print(np.linalg.inv(tmp))