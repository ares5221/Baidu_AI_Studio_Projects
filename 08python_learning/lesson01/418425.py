#!/usr/bin/env python
# coding: utf-8

# ### 作业一：输出 9*9 乘法口诀表(注意格式)
# 
# 注意：提交作业时要有代码执行输出结果。

# In[2]:


def table():
    for i in range(1, 10):
        for j in range(1, i + 1):
            print(f"{j}*{i}={i * j}", end="\t")
        print(" ")

if __name__ == '__main__':
    table()


# ### 作业二：查找特定名称文件
# 遍历”Day1-homework”目录下文件；
# 
# 找到文件名包含“2020”的文件；
# 
# 将文件名保存到数组result中；
# 
# 按照序号、文件名分行打印输出。
# 
# 注意：提交作业时要有代码执行输出结果。

# In[5]:


#导入OS模块
import os
#待搜索的目录路径
path = "./Day1-homework"
#待搜索的名称
filename = "2020"
#定义保存结果的数组
result = []

def findfiles():
    #在这里写下您的查找文件代码吧！
    for root, dirs, files in os.walk(path):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # # 遍历所有的文件夹
        # for d in dirs:
        #     print(os.path.join(root, d))
        # 遍历文件
        for f in files:
            if filename in f:
                result.append(os.path.join(root,f))

    for idx in range(len(result)):
        print([idx+1,result[idx]])

if __name__ == '__main__':
    findfiles()

