#!/usr/bin/env python
# coding: utf-8

# # 综合大作业
# 
# **第一步：爱奇艺《青春有你2》评论数据爬取**(参考链接：[https://www.iqiyi.com/v_19ryfkiv8w.html#curid=15068699100_9f9bab7e0d1e30c494622af777f4ba39](http://))
# * 爬取任意一期正片视频下评论
# * 评论条数不少于1000条
# 
# **第二步：词频统计并可视化展示**
# * 数据预处理：清理清洗评论中特殊字符（如：@#￥%、emoji表情符）,清洗后结果存储为txt文档
# * 中文分词：添加新增词（如：青你、奥利给、冲鸭），去除停用词（如：哦、因此、不然、也好、但是）
# * 统计top10高频词
# * 可视化展示高频词
# 
# **第三步：绘制词云**
# * 根据词频生成词云
# * 可选项-添加背景图片，根据背景图片轮廓生成词云 
# 
# **第四步：结合PaddleHub，对评论进行内容审核**
# 

# # 需要的配置和准备
# * 中文分词需要jieba
# * 词云绘制需要wordcloud
# * 可视化展示中需要的中文字体
# * 网上公开资源中找一个中文停用词表
# * 根据分词结果自己制作新增词表
# * 准备一张词云背景图（附加项，不做要求，可用hub抠图实现）
# * paddlehub配置
# 
# 

# In[2]:


# get_ipython().system('pip install jieba')
# get_ipython().system('pip install wordcloud')


# In[ ]:





# In[3]:


# Linux系统默认字体文件路径
# !ls /usr/share/fonts/
# 查看系统可用的ttf格式中文字体
# get_ipython().system('fc-list :lang=zh | grep ".ttf"')


# In[ ]:


# !wget https://mydueros.cdn.bcebos.com/font/simhei.ttf # 下载中文字体
# #创建字体目录fonts
# !mkdir .fonts
# # 复制字体文件到该路径
# !cp simhei.ttf .fonts/


# In[ ]:


#安装模型
# !hub install porn_detection_lstm==1.1.0
# !pip install --upgrade paddlehub


# In[4]:


from __future__ import print_function
import requests
import json
import re #正则匹配
import time #时间处理模块
import jieba #中文分词
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from PIL import Image
from wordcloud import WordCloud  #绘制词云模块
import paddlehub as hub


# In[ ]:


#请求爱奇艺评论接口，返回response信息
def getMovieinfo(url):
    '''
    请求爱奇艺评论接口，返回response信息
    参数  url: 评论的url
    :return: response信息
    '''
    sesson = requests.session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }
    response = sesson.get(url,headers=headers)
    if response.status_code ==200:
        return response.text
    return None

#解析json数据，获取评论
def saveMovieInfoToFile(lastID,arr):
    '''
    解析json数据，获取评论
    参数  lastId:最后一条评论ID  arr:存放文本的list
    :return: 新的lastId
    '''
    url = 'https://sns-comment.iqiyi.com/v3/comment/get_comments.action?agent_type=118&' \
          'agent_version=9.11.5&authcookie=null&business_type=17&' \
          'content_id=15068699100&page=&page_size=20&types=time&hot_size=0&last_id='
    # 240904557221&callback=jsonp_1588067183227_32803
    url += str(lastID)
    responseTxt = getMovieinfo(url)
    responseJson = json.loads(responseTxt)
    comments = responseJson['data']['comments']
    for val in comments:
        if 'content' in val.keys():
            print(val['content'])
            arr.append(val['content'])
        lastID = str(val['id'])
    return lastID


# In[ ]:


#去除文本中特殊字符
def clear_special_char(content):
    '''
    正则处理特殊字符
    参数 content:原文本
    return: 清除后的文本
    '''
    s = re.sub(r'</?(.+?)>|&nbsp;|\t|\r','',content)
    s = re.sub(r'\n','',s)
    s = re.sub(r'\*','\\*',s)
    s = re.sub('[^\u4e00-\u9fa5^a-z^A-Z^0-9]','',s)
    s = re.sub('[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f]', '', s)
    s = re.sub('[a-zA-Z]', '', s)
    s = re.sub('^\d+(\.\d+)?$', '\\*', s)
    return s


# In[ ]:


def fenci(text):
    '''
    利用jieba进行分词
    参数 text:需要分词的句子或文本
    return：分词结果
    '''
    jieba.load_userdict('add_words.txt')
    seg = jieba.lcut(text,cut_all=False)
    
    return seg


# In[ ]:


def stopwordslist(file_path):
    '''
    创建停用词表
    参数 file_path:停用词文本路径
    return：停用词list
    '''
    stopwords = [line.strip() for line in open(file_path,encoding='utf-8').readlines()]
    return stopwords


# In[ ]:


def movestopwords(sentences,stopwords,counts):
    '''
    去除停用词,统计词频
    参数 file_path:停用词文本路径 stopwords:停用词list counts: 词频统计结果
    return：None
    '''
    out = []
    for word in sentences:
        if word not in stopwords:
            if len(word) !=1:
                counts[word] = counts.get(word,0) +1
    return None
    


# In[ ]:


def drawcounts(counts,num):
    '''
    绘制词频统计表
    参数 counts: 词频统计结果 num:绘制topN
    return：none
    '''
    x_aixs = []
    y_aixs = []
    c_order = sorted(counts.items(),key=lambda x:x[1],reverse=True)
    for c in c_order[:num]:
        x_aixs.append(c[0])
        y_aixs.append(c[1])

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.bar(x_aixs,y_aixs)
    plt.title('词频统计结果')
    plt.show()
    return
    


# In[ ]:


def drawcloud(word_f):
    '''
    根据词频绘制词云图
    参数 word_f:统计出的词频结果
    return：none
    '''
    # 加载背景图片
    cloud_mask = np.array(Image.open('cloud.png'))
    #忽略显示的词
    st = set(['东西','这是'])
    # 生成wordcloud对象
    wc = WordCloud(background_color='white',
                   mask=cloud_mask,
                   max_words=150,
                   font_path='simhei.ttf',
                   min_font_size=10,
                   max_font_size=100,
                   width=400,
                   relative_scaling=0.3,
                   stopwords=st)
    wc.fit_words(word_f)
    wc.to_file('pic.png')


# In[ ]:


def text_detection(test_text,file_path):
    '''
    使用hub对评论进行内容分析
    return：分析结果

    '''
    pron_detection_lstm = hub.Module(name='porn_detection_lstm')
    f = open(file_path,'r',encoding='utf-8')
    for line in f:
        if len(line.strip()) ==1:
            continue
        else:
            test_text.append(line)
    f.close()
    input_dict = {'text':test_text}
    results = pron_detection_lstm.detection(data=input_dict,use_gpu=True,batch_size=1)
    for index,item in enumerate(results):
        if item['porn_detection_key'] =='porn':
            print(item['text'],':',item['porn_probs'])
    


# In[ ]:


#评论是多分页的，得多次请求爱奇艺的评论接口才能获取多页评论,有些评论含有表情、特殊字符之类的
#num 是页数，一页10条评论，假如爬取1000条评论，设置num=100
if __name__ == "__main__":
    num = 100
    lastId = '0'
    arr = []
    # with open('aqy.txt','a',encoding='utf-8') as f:
    #     for i in range(num):
    #         lastId = saveMovieInfoToFile(lastId,arr)
    #         time.sleep(0.5)
    #     for item in arr:
    #         Item = clear_special_char(item)
    #         if Item.strip() !='':
    #             try:
    #                 f.write(Item+'\n')
    #             except Exception as e:
    #                 print('含有特殊字符')
    # print('共有评论',len(arr))
    f = open('aqy.txt','r',encoding='utf-8')
    counts = {}
    for line in f:
        words = fenci(line)
        stopwords = stopwordslist('cn_stopwords.txt')
        movestopwords(words,stopwords,counts)
    drawcounts(counts,10)
    drawcloud(counts)
    f.close()

    file_path = 'aqy.txt'
    test_text = []
    # text_detection(test_text,file_path)



# In[ ]:


# display(Image.open('pic.png')) #显示生成的词云图像

