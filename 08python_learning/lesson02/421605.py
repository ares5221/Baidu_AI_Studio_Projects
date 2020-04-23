#!/usr/bin/env python
# coding: utf-8

# # ！！！作业说明！！！
# 
# ### 1.请在下方提示位置，补充代码，完成《青春有你2》选手图片爬取，将爬取图片进行保存，保证代码正常运行
# ### 2.打印爬取的所有图片的绝对路径，以及爬取的图片总数，此部分已经给出代码。请在提交前，一定要保证有打印结果，如下图所示：
# <img src="https://ai-studio-static-online.cdn.bcebos.com/bc57d8895ede4f1f8407524fa0bdab305f55823a28ac42b69a07f56787a779c0" width='500' height='500'>

# <br/>
# <br/>
# 
# **深度学习一般过程:**
# <br/>
# <br/>
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/b372f0b8277a40759b91920972951f1184bca6c33fc74881bd4f93f7388ed32a" width='600' height='600'/>
# 
# 
# <br/>
# <br/>
# <br/>
# 
# **收集数据，尤其是有标签、高质量的数据是一件昂贵的工作。**
# 
# <br/>
# 
# **爬虫**的过程，就是模仿浏览器的行为，往目标站点发送请求，接收服务器的响应数据，提取需要的信息，并进行保存的过程。
# 
# **Python**为爬虫的实现提供了工具:requests模块、BeautifulSoup库
# 
# 

# ## 任务描述
# <br/>
# 
# **本次实践使用Python来爬取百度百科中《青春有你2》所有参赛选手的信息。**
# 
# 数据获取：https://baike.baidu.com/item/青春有你第二季
# 
# <img src ="https://ai-studio-static-online.cdn.bcebos.com/6e0058bd57084dec989b2dc68bd61c13d2ad8d4532714aa2a40b6b9818da9037" height='500' width='500'/>
# <img src="https://ai-studio-static-online.cdn.bcebos.com/e6b2850eabaf4e40882d35d639a81ba1481dbbb212e443979f91fa39bc416117" height='500' width='500' />
# 
# 
# <br/>
# <br/>
# 

# <br/>
# 
# **上网的全过程:**
# 
#     普通用户:
# 
#     打开浏览器 --> 往目标站点发送请求 --> 接收响应数据 --> 渲染到页面上。
# 
#     爬虫程序:
# 
#     模拟浏览器 --> 往目标站点发送请求 --> 接收响应数据 --> 提取有用的数据 --> 保存到本地/数据库。
# 
# 
# **爬虫的过程**：
# 
#     1.发送请求（requests模块）
# 
#     2.获取响应数据（服务器返回）
# 
#     3.解析并提取数据（BeautifulSoup查找或者re正则）
# 
#     4.保存数据
# 
# 

# 
# <br/>
# 
# **本实践中将会使用以下两个模块，首先对这两个模块简单了解以下：**

# <br/>
# 
# **request模块：**
# 
#     requests是python实现的简单易用的HTTP库，官网地址：http://cn.python-requests.org/zh_CN/latest/
#     
#     requests.get(url)可以发送一个http get请求，返回服务器响应内容。
#     
#     
# 
# 
# 
# 

# <br/>
# 
# **BeautifulSoup库：**
# 
#     BeautifulSoup 是一个可以从HTML或XML文件中提取数据的Python库。网址：https://beautifulsoup.readthedocs.io/zh_CN/v4.4.0/
#     
#     BeautifulSoup支持Python标准库中的HTML解析器,还支持一些第三方的解析器,其中一个是 lxml。
#     
#     BeautifulSoup(markup, "html.parser")或者BeautifulSoup(markup, "lxml")，推荐使用lxml作为解析器,因为效率更高。

# In[1]:


#如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
#!mkdir /home/aistudio/external-libraries
#!pip install beautifulsoup4 -t /home/aistudio/external-libraries
#!pip install lxml -t /home/aistudio/external-libraries


# In[2]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
import sys
sys.path.append('/home/aistudio/external-libraries')


# ## 一、爬取百度百科中《青春有你2》中所有参赛选手信息，返回页面数据

# In[3]:


import json
import re
import requests
import datetime
from bs4 import BeautifulSoup
import os

#获取当天的日期,并进行格式化,用于后面文件命名，格式:20200420
today = datetime.date.today().strftime('%Y%m%d')    

def crawl_wiki_data():
    """
    爬取百度百科中《青春有你2》中参赛选手信息，返回html
    """
    headers = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }
    url='https://baike.baidu.com/item/青春有你第二季'                         

    try:
        response = requests.get(url,headers=headers)
        print(response.status_code)

        #将一段文档传入BeautifulSoup的构造方法,就能得到一个文档的对象, 可以传入一段字符串
        soup = BeautifulSoup(response.text,'lxml')
        
        #返回的是class为table-view log-set-param的<table>所有标签
        tables = soup.find_all('table',{'class':'table-view log-set-param'})

        crawl_table_title = "参赛学员"

        for table in  tables:           
            #对当前节点前面的标签和字符串进行查找
            table_titles = table.find_previous('div').find_all('h3')
            for title in table_titles:
                if(crawl_table_title in title):
                    return table       
    except Exception as e:
        print(e)



# ## 二、对爬取的页面数据进行解析，并保存为JSON文件

# In[4]:


def parse_wiki_data(table_html):
    '''
    从百度百科返回的html中解析得到选手信息，以当前日期作为文件名，存JSON文件,保存到work目录下
    '''
    bs = BeautifulSoup(str(table_html),'lxml')
    all_trs = bs.find_all('tr')

    error_list = ['\'','\"']

    stars = []

    for tr in all_trs[1:]:
         all_tds = tr.find_all('td')

         star = {}

         #姓名
         star["name"]=all_tds[0].text
         #个人百度百科链接
         star["link"]= 'https://baike.baidu.com' + all_tds[0].find('a').get('href')
         #籍贯
         star["zone"]=all_tds[1].text
         #星座
         star["constellation"]=all_tds[2].text
         #身高
         star["height"]=all_tds[3].text
         #体重
         star["weight"]= all_tds[4].text

         #花语,去除掉花语中的单引号或双引号
         flower_word = all_tds[5].text
         for c in flower_word:
             if  c in error_list:
                 flower_word=flower_word.replace(c,'')
         star["flower_word"]=flower_word 
         
         #公司
         if not all_tds[6].find('a') is  None:
             star["company"]= all_tds[6].find('a').text
         else:
             star["company"]= all_tds[6].text  

         stars.append(star)

    json_data = json.loads(str(stars).replace("\'","\""))   
    with open('work/' + today + '.json', 'w', encoding='UTF-8') as f:
        json.dump(json_data, f, ensure_ascii=False)


# ## 三、爬取每个选手的百度百科图片，并进行保存

# ## ！！！请在以下代码块中补充代码，爬取每个选手的百度百科图片，并保存 ！！！

# In[5]:


def crawl_pic_urls():
    '''
    爬取每个选手的百度百科图片，并保存
    ''' 
    with open('work/'+ today + '.json', 'r', encoding='UTF-8') as file:
         json_array = json.loads(file.read())

    headers = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36' 
     }

    for star in json_array:
        name = star['name']
        link = star['link']
        pic_urls = []
        #！！！请在以下完成对每个选手图片的爬取，将所有图片url存储在一个列表pic_urls中！！！
        try:
            response = requests.get(link, headers=headers)
            soup = BeautifulSoup(response.text, 'lxml')
            imgs = soup.find_all('img')

            for img in imgs:
                src = img.get('src')
                if src.startswith('http'):
                    pic_urls.append(src)
        except Exception as e:
            print(e)
        #！！！根据图片链接列表pic_urls, 下载所有图片，保存在以name命名的文件夹中！！！
        down_pic(name,pic_urls)

     


# In[6]:


def down_pic(name,pic_urls):
    '''
    根据图片链接列表pic_urls, 下载所有图片，保存在以name命名的文件夹中,
    '''
    path = 'work/'+'pics/'+name+'/'

    if not os.path.exists(path):
      os.makedirs(path)

    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            string = str(i + 1) + '.jpg'
            with open(path+string, 'wb') as f:
                f.write(pic.content)
                print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue


# ## 四、打印爬取的所有图片的路径

# In[7]:


def show_pic_path(path):
    '''
    遍历所爬取的每张图片，并打印所有图片的绝对路径
    '''
    pic_num = 0
    for (dirpath,dirnames,filenames) in os.walk(path):
        for filename in filenames:
           pic_num += 1
           print("第%d张照片：%s" % (pic_num,os.path.join(dirpath,filename)))           
    print("共爬取《青春有你2》选手的%d照片" % pic_num)
    


# In[8]:


if __name__ == '__main__':

     #爬取百度百科中《青春有你2》中参赛选手信息，返回html
     html = crawl_wiki_data()

     #解析html,得到选手信息，保存为json文件
     parse_wiki_data(html)

     #从每个选手的百度百科页面上爬取图片,并保存
     crawl_pic_urls()

     #打印所爬取的选手图片路径
     show_pic_path('./work/pics/')

     print("所有信息爬取完成！")

