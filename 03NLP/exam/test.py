#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import random
names = ["财经", "彩票", "房产", "股票", "家居", "教育", "科技",
                    "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]
with open('result.txt','a',encoding='utf-8') as f:
    for i in range(83559):
        ss = random.choice(names)
        print(ss)
        f.write(ss+'\n')
