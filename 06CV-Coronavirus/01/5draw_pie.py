#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.faker import Faker
import json
import datetime

# 读原始数据文件
today = datetime.date.today().strftime('%Y%m%d')   #20200315
datafile = 'data/'+ today + '.json'
with open(datafile, 'r', encoding='UTF-8') as file:
    json_array = json.loads(file.read())

# 分析全国实时确诊数据：'confirmedCount'字段
china_data = []
for province in json_array:
    china_data.append([province['provinceShortName'], province['confirmedCount']])
china_data = sorted(china_data, key=lambda x: x[1], reverse=True)                 #reverse=True,表示降序，反之升序

print(china_data)

c = (
    Pie()
    .add("", china_data, center=[550,360], radius=[0,120])
    .set_global_opts(title_opts=opts.TitleOpts(title="20200331全国疫情分布图"))
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"), pos_left="80%", orient="vertical")
    .render("pie_base.html")
)
