#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
#  {'虞书欣': 0, '许佳琪': 1, '赵小棠': 2, '安崎': 3, '王承渲': 4}
print(os.listdir('.'))
for dir in os.listdir('.'):
    print(dir)
    if not (dir.endswith('.py') or dir.endswith('.txt')):
        with open('train_list.txt','a',encoding='utf-8') as f:
            if dir =='yu':
                label = 0
            elif dir =='xu':
                label =1
            elif dir =='zhao':
                label =2
            elif dir =='an':
                label =3
            elif dir == 'wang':
                label =4
            else:
                pass
            for tmp in os.listdir('./'+dir):
                print('dataset/'+dir +'/'+tmp, label)
                f.write(dir +'/'+tmp +' ' + str(label))
                f.write('\n')