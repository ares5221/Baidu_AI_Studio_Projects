#!/usr/bin/env python
# coding: utf-8

# ## ！！！作业说明！！！
# 
# ## 请在下方提示位置，补充代码，对《青春有你2》对选手体重分布进行可视化，绘制饼状图，如下图所示：
# 
# ## 不要求跟下图样式一模一样
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/6c64e404049f46d8830d0b275682faec75cb88f38b494b308146ace40a9e1743" height=500 width=500>
# 

# ## 任务描述：
# 
# 基于第二天实践使用Python来爬去百度百科中《青春有你2》所有参赛选手的信息，进行数据可视化分析。
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/a1fc9f2d97a644c2b5409da939fded8639bcebf71a2c42d1a4a01d0116d1783e" height='600' width='600'/>
# 
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/31c9f4f3e7e640fdbff7264d61c01f62a38419a990a84c898587f3ace818128a" height='600' width='600' />
# 
# 

# In[15]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
#!mkdir /home/aistudio/external-libraries
#!pip install matplotlib -t /home/aistudio/external-libraries


# In[16]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')


# In[17]:


# 下载中文字体
# get_ipython().system('wget https://mydueros.cdn.bcebos.com/font/simhei.ttf')
# 将字体文件复制到matplotlib字体路径
# get_ipython().system('cp simhei.ttf /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/')
# 一般只需要将字体文件复制到系统字体目录下即可，但是在aistudio上该路径没有写权限，所以此方法不能用
# !cp simhei.ttf /usr/share/fonts/

# 创建系统字体文件路径
# get_ipython().system('mkdir .fonts')
# 复制文件到该路径
# get_ipython().system('cp simhei.ttf .fonts/')
# get_ipython().system('rm -rf .cache/matplotlib')


# ## 绘制选手区域分布柱状图

# In[18]:


import matplotlib.pyplot as plt
import numpy as np 
import json
import matplotlib.font_manager as font_manager

#显示matplotlib生成的图形
# get_ipython().run_line_magic('matplotlib', 'inline')

with open('data/data31557/20200422.json', 'r', encoding='UTF-8') as file:
         json_array = json.loads(file.read())

#绘制小姐姐区域分布柱状图,x轴为地区，y轴为该区域的小姐姐数量

zones = []
for star in json_array:
    zone = star['zone']
    zones.append(zone)
print(len(zones))
print(zones)


zone_list = []
count_list = []

for zone in zones:
    if zone not in zone_list:
        count = zones.count(zone)
        zone_list.append(zone)
        count_list.append(count)

print(zone_list)
print(count_list)

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体

plt.figure(figsize=(20,15))

plt.bar(range(len(count_list)), count_list,color='r',tick_label=zone_list,facecolor='#9999ff',edgecolor='white')

# 这里是调节横坐标的倾斜度，rotation是度数，以及设置刻度字体大小
plt.xticks(rotation=45,fontsize=20)
plt.yticks(fontsize=20)

plt.legend()
plt.title('''《青春有你2》参赛选手''',fontsize = 24)
plt.savefig('./work/result/bar_result.png')
plt.show()


# In[19]:


import matplotlib.pyplot as plt
import numpy as np 
import json
import matplotlib.font_manager as font_manager
import pandas as pd

#显示matplotlib生成的图形
# get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_json('data/data31557/20200422.json')
#print(df)

grouped=df['name'].groupby(df['zone'])
s = grouped.count()

zone_list = s.index
count_list = s.values


# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体

plt.figure(figsize=(20,15))

plt.bar(range(len(count_list)), count_list,color='r',tick_label=zone_list,facecolor='#9999ff',edgecolor='white')

# 这里是调节横坐标的倾斜度，rotation是度数，以及设置刻度字体大小
plt.xticks(rotation=45,fontsize=20)
plt.yticks(fontsize=20)

plt.legend()
plt.title('''《青春有你2》参赛选手''',fontsize = 24)
plt.savefig('./work/result/bar_result02.png')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.font_manager as font_manager
import pandas as pd

#显示matplotlib生成的图形

df = pd.read_json('data/data31557/20200422.json')
#print(df)
weight_list=df['weight']
info = [0 for i in range(4)]
w_name = ['<=45kg','45-50kg','50-55kg','>55kg']
for wl in weight_list:
    curr = float(wl[:-2])
    if curr <=45:
        info[0] +=1
    elif curr>45 and curr<=50:
        info[1]+=1
    elif curr >50 and curr<=55:
        info[2] +=1
    else:
        info[3] +=1

print(info)

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体

plt.figure(figsize=(20,15))
explode =[0.05,0,0.01,0.01]
plt.pie(info,explode=explode,labels=w_name,autopct='%1.1f%%',shadow=False,startangle=150,textprops= {'fontsize':24,'color':'black'})


plt.legend()
plt.title('''《青春有你2》参赛选手''',fontsize = 24)
plt.savefig('./work/result/pie.png')
plt.show()




