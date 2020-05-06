#!/usr/bin/env python
# coding: utf-8

# # PaddleHub创意赛：四个视频合成~大尺寸AI美女风景图 20200408再次升级!
# 
# # 增加视频处理功能!
# 
# # 先看效果这里都转成了GIF
# 
# 原始视频：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/70f7ed56a0da40868bff328b59203650764949f626b243bea348dfd9c06b47cf)
# 
# 抠图合并后：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/f9b92639b16e4181a40e65ecee34ca870570159d09a44d7e834574e386b60ae0)
# 
# 还尝试了多种模型例如：
# 
# stylepro_artistic
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/362555b9e3904f44800e351f9b0b1e1b04b74ad7970546a4aa2be3dea45486cc)
# 
# attgan_celeba
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/e987851728d54da3b715f4e0cc06d47030ac4fb6d1e54a8292f7f5a0d55e9a0f)
# 
# sace2p
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/168adbcb13094709b118ca665d4701072570a1684461463e91d105384ec33f74)
# 
# cyclegan_cityscapes
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/f2f2b4730ed94e3bb28db3b034757b5e57fe8e49e7a142dbb46efc62b5d6ca50)
# 
# 检测口罩模型：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/5b5d51dab33c44b1b2c5ad163d7d7951669c5f2862e54f51bbc33653eaf73374)
# 
# 口罩识别模型也可以自己训练，有相应课程和项目。
# 
# 上面有效果不好的，因为那是对抗模型生成滴。
# 
# 然后是合集：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/c9cc3bb118ef4513a010afb78ba2eea2697ccca72fc24cd1a0e5a06c0bf4a70a)
# 
# 
# 直接看最下面啦!
# 
# 本示例用DeepLabv3+模型完成一键抠图。在最新作中，作者通过encoder-decoder进行多尺度信息的融合，同时保留了原来的空洞卷积和ASSP层， 其骨干网络使用了Xception模型，提高了语义分割的健壮性和运行速率，在 PASCAL VOC 2012 dataset取得新的state-of-art performance，该PaddleHub Module使用百度自建数据集进行训练，可用于人像分割，支持任意大小的图片输入。在完成一键抠图之后，通过图像合成，实现扣图比赛任务
# 
# 更多PaddleHub预训练模型应用可见：[教程合集课程](http://aistudio.baidu.com/aistudio/course/introduce/1070)
# 
# PaddleHub抠图比赛会在4月1号晚七点在b站直播，可以先行关注飞桨b站公众号：[飞桨PaddlePaddle](http://https://space.bilibili.com/476867757?from=search&seid=6064675744229842869)
# 
# **NOTE：** 如果您在本地运行该项目示例，需要首先安装PaddleHub。如果您在线运行，需要首先fork该项目示例。之后按照该示例操作即可。
# 
# 

# ## 一、定义待抠图照片
# 
# 
# 以本示例中文件夹下meditation.jpg为待预测图片

# In[1]:


# 解压准备好的压缩文件
#!unzip /home/aistudio/data/data27839/background_pic
get_ipython().system('unzip -o work/background_pic.zip')
get_ipython().system('unzip -o work/front_pic.zip')


# In[2]:


get_ipython().system('pip install paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple')


# # 1.待预测图片
# test_img_path = ["/home/aistudio/"+"work/front_pic/model-beautiful-women-s-people-1cfb484b3fb353f0f117b9c34a31e45b.jpg"]
# 
# import matplotlib.pyplot as plt 
# import matplotlib.image as mpimg 
# 
# #img = mpimg.imread(test_img_path[0]) 
# 
# # 展示待预测图片
# #plt.figure(figsize=(10,10))
# #plt.imshow(img) 
# #plt.axis('off') 
# #plt.show()

# ## 二、加载预训练模型
# 
# 通过加载PaddleHub DeepLabv3+模型(deeplabv3p_xception65_humanseg)实现一键抠图

# In[9]:


import paddlehub as hub
test_img_path = '/home/aistudio/humanseg_output/" + "model-beautiful-women-s-people-1cfb484b3fb353f0f117b9c34a31e45b.png'

module = hub.Module(name="deeplabv3p_xception65_humanseg")
input_dict = {"image": test_img_path}

# execute predict and print the result
results = module.segmentation(data=input_dict)
for result in results:
    print(result)

# 预测结果展示
#test_img_path = "./humanseg_output/meditation.jpg"
#img = mpimg.imread(test_img_path)
#plt.figure(figsize=(10,10))
#plt.imshow(img) 
#plt.axis('off') 
#plt.show()


# ## 三、图像合成
# 
# 将抠出的人物图片合成在想要的背景图片当中。
# 

# In[10]:


from PIL import Image
import numpy as np

def blend_images(fore_image, base_image):
    """
    将抠出的人物图像换背景
    fore_image: 前景图片，抠出的人物图片
    base_image: 背景图片
    """
    # 读入图片
    base_image = Image.open(base_image).convert('RGB')
    fore_image = Image.open(fore_image).resize(base_image.size)

    # 图片加权合成
    scope_map = np.array(fore_image)[:,:,-1] / 255
    scope_map = scope_map[:,:,np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[:,:,:3]) + np.multiply((1-scope_map), np.array(base_image))
    
    #保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save("blend_res_img.jpg")

    


# In[11]:


blend_images(test_img_path , 'sea_wave.jpg')

# 展示合成图片
plt.figure(figsize=(10,10))
img = mpimg.imread("./blend_res_img.jpg")
plt.imshow(img) 
plt.axis('off') 
plt.show()


# # 新内容藏这里....以上是原版的内容，经过两天的Py学习，掌握了基本使用方法和用GPU的跑法。具体看 work/mixture_beauty.py 文件，内容根据自己需要求改。其他的文件work/unzipfile.py,work/koutu.py,work/hebin.py是对work/mixture_beauty.py 的拆解，因为跑的时间比较长。。所以可以在跑一部分的时候，编写下一部分程序，调试，不浪费时间，最后再整合。
# 

# 效果如下：
# ![](https://ai-studio-static-online.cdn.bcebos.com/b7179236a6ca4d4682cc4b6b13e361ec5729d77cacea4329997d8da0161ddbaa)
# 

# # 这是20200406新增内容,如何抠视频!
# 
# 我们知道视频是一帧一帧的图像合成的,那么根据把视频拆分成图像不就可以了嘛.
# 测试文件:test_video.mp4
# 
# 注意看属性:
# ![](https://ai-studio-static-online.cdn.bcebos.com/93c98db57f094c9f855c1a62d796b6ec6f01406269a743c3b30ae6bd2cf03edd)
# 24帧每秒,记下来,后面要用.
# 下面我们使用函数来分解视频:

# In[12]:


#先升级
get_ipython().system('pip install --upgrade paddlehub -i https://mirror.baidu.com/pypi/simple')


# In[16]:


#加载库
from PIL import Image
from sklearn.model_selection import train_test_split
import datetime
import hashlib
import json
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
import os
import paddle
import paddle.fluid as fluid
import paddlehub
import rarfile
import requests
import shutil
import six
import sys
import tarfile
import time
import zipfile
import msg
import subprocess
import cv2
from natsort import natsorted
import random


# In[17]:


#视频分帧
def video_to_images(file_name,filepath): #路径要带 '/'
    import cv2
    vidcap = cv2.VideoCapture(file_name)
    success,image = vidcap.read()
    count = 0
    success = True
    angle=90
    while success:
        success,image = vidcap.read()
        if success==True:
            cv2.imwrite(filepath + "frame%d.jpg" % count, image)   # save frame as JPEG file
            # if cv2.waitKey(10) == 27:  #用户按下ESC(ASCII码为27),则跳出循环  
            #     break
            count = count + 1


# In[18]:


#调用方式   把"work/test_video.mp4"视频,分解到"work/video2image"目录 路径要带 '/'
video_to_images("work/test_video.mp4","work/video2image/")


# In[19]:


# 然后抠图,之前的mixture_beauty.py中也用到了循环抠图,这里写成了函数.注意带路径的都要带 '/',这里都是要传入路径的
def koutu(fore_image_path,paddlehub_out_path): 
# 1.指定待预测图片路径，遍历循环抠图 保存到 paddlehub_out_path
    g = os.walk(fore_image_path)
    for path,dir_list,file_list in g: 
        file_list = natsorted(file_list)
        for file_name in file_list:
            #取出一个前景文件
            front_pic = [os.path.join(path, file_name)]
            # print (front_pic)
            # 2.加载预训练模型
            # 通过加载PaddleHub DeepLabv3+模型(deeplabv3p_xception65_humanseg)实现一键抠图
            module = paddlehub.Module(name="deeplabv3p_xception65_humanseg")
            input_dict = {"image": front_pic}
            #print (input_dict)
            #GPU开启下面一行,如果用下面一行，必须安装paddlepaddle-gpu
            results = module.segmentation(data=input_dict,use_gpu = gpu,batch_size = 10,output_dir=paddlehub_out_path)
            #为提高效率,不再输出结果
            #for result in results: 
            #print(result)


# ### 注意这一行 results = module.segmentation(data=input_dict,use_gpu = True,batch_size = 10,output_dir=paddlehub_out_path)
# ### 可以学习到三部分:
# ### use_gpu = True 是否使用GPU
# ### batch_size = 10 如果是GPU,助教说可以设置成128,256,512..提高效率,大家可以试试
# ### output_dir=paddlehub_out_path 输出的路径,注意,不是所有模型都支持这个参数.
# 

# In[20]:


#抠图后再合并,支持前景图和背景图都是同一个目录

#俩图片合并函数
def blend_images(fore_image, base_image,output_path):
    """
    将抠出的人物图像换背景
    fore_image: 前景图片，抠出的人物图片
    base_image: 背景图片
    """
    # 读入图片 取出不含路径的文件名
    houtput_pic =output_path + os.path.basename(fore_image)+"_"+ os.path.basename(base_image)
    base_image = Image.open(base_image).convert('RGB')
    fore_image = Image.open(fore_image).resize(base_image.size)

    # 图片加权合成
    scope_map = np.array(fore_image)[:,:,-1] / 255
    scope_map = scope_map[:,:,np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[:,:,:3]) + np.multiply((1-scope_map), np.array(base_image))
    
    #保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save(houtput_pic)
    # print (houtput_pic)

#俩目录合并

def hebing_images(fore_image_path, base_image_path,output_path):
    """
    将抠出的人物图像换背景
    fore_image_path: 前景图片路径，抠出的人物图片
    base_image_path: 背景图片路径
    """
    f = os.walk(fore_image_path)  
    for path,dir_list,file_list in f:
        file_list = natsorted(file_list)    
        for file_name in file_list:  
            humanseg_pic =os.path.join(path, file_name)
            #图像操作 背景路径循环
            b = os.walk(base_image_path)  
            for path1,dir_list1,file_list1 in b:  
                for file_name1 in file_list1:
                    background_pic = os.path.join(path1, file_name1)
                #合成
                    #print(humanseg_pic , background_pic)
                    blend_images(humanseg_pic , background_pic,output_path)


# 图片经处理后会有大小可能会有变化，如果要做4合一视频，需要把每段视频都调整成相同分辨率。
# 然后合并，这里我先使用了命令行工具，之后再用Python实现一次。

# In[21]:


#多帧合成视频 ,下面fps = 25 这个25就是之前的要 记录下的帧数
def images_to_video(input_filepath,output_filename): 
    fps = 25 # 帧率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # img_array = []
    g = os.walk(input_filepath)
    count =0 
    for path,dir_list,file_list in g: 
        # file_list = natsorted(file_list)
        for file_name in file_list:
            #取出一个前景文件
            front_pic = os.path.join(path, file_name)
            # print (front_pic)
            img=cv2.imread(front_pic)
            #此处应该只取一次属性
            imgInfo = img.shape
            size = (imgInfo[1],imgInfo[0])
            #获取图片宽高度信息
            #此处应该只取一次属性
            while count <= 1:
                count = count + 1
                img=cv2.imread(front_pic)
                imgInfo = img.shape
                size = (imgInfo[1],imgInfo[0])
                out = cv2.VideoWriter(output_filename, fourcc, fps,size) 
            if img is None:
                print(front_pic + " is non-existent!")
                continue
            else:
                out.write(img)
    # out.release()



# 下面是用到的模型，更改路径即可批量处理。可以看到，每个模型调用基本命令都相同，只需要更改很少地方。当然，有的模型不支持输出路径，有的还有特别参数，具体看官方文档。以实际为准。

# In[22]:


def stylepro_artistic(fore_image_path,paddlehub_out_path): 
# 1.指定待预测图片路径，遍历循环抠图 保存到 paddlehub_out_path
    g = os.walk(fore_image_path)
    for path,dir_list,file_list in g: 
        file_list = natsorted(file_list)
        for file_name in file_list:
            #取出一个前景文件
            
            dstfile=os.path.join("/home/aistudio/work/3boys_front_bak/", file_name)

            front_pic = os.path.join(path, file_name)
            # print (front_pic)
            # 2.加载预训练模型
            module = paddlehub.Module(name="stylepro_artistic")
            input_dict = {"content": front_pic,"styles":["/home/aistudio/work/Picasso_pic.jpg"]}
            #print (input_dict)
            #GPU开启下面一行,如果用下面一行，必须安装paddlepaddle-gpu
            results = module.style_transfer(paths=[input_dict],use_gpu = True,visualization=True,output_dir=paddlehub_out_path)
            mymovefile(front_pic,dstfile)
            #以下还可以优化吗
            #for result in results: 
            #print(result)


# In[23]:


def attgan_celeba(fore_image_path,paddlehub_out_path): 
# 待处理图片尽量只露脸，当五官是朝向正前方且露出五官时，效果会比较好。

# 待处理图片的尺寸接近 128 * 128 像素时，效果会比较好。
# 1.指定待预测图片路径，遍历循环抠图 保存到 HUMANSEG_SPACE
    g = os.walk(fore_image_path)
    for path,dir_list,file_list in g: 
        file_list = natsorted(file_list)
        for file_name in file_list:
            #取出一个前景文件
            front_pic = os.path.join(path, file_name)
            # print (front_pic)
            # 2.加载预训练模型
            # 通过加载PaddleHub DeepLabv3+模型(deeplabv3p_xception65_humanseg)实现一键抠图
            module = paddlehub.Module(name="attgan_celeba")
            input_dict = {"image": [front_pic],"style":["Eyeglasses"]}
            #print (input_dict)
            #GPU开启下面一行,如果用下面一行，必须安装paddlepaddle-gpu
            results = module.generate(data=input_dict,use_gpu = gpu,output_dir=paddlehub_out_path)
            #以下还可以优化吗
            #for result in results: 
            #print(result)


# In[ ]:


def ace2p(fore_image_path,paddlehub_out_path): 
# 待处理图片尽量只露脸，当五官是朝向正前方且露出五官时，效果会比较好。

# 待处理图片的尺寸接近 128 * 128 像素时，效果会比较好。
# 1.指定待预测图片路径，遍历循环抠图 保存到 HUMANSEG_SPACE
    g = os.walk(fore_image_path)
    for path,dir_list,file_list in g: 
        file_list = natsorted(file_list)
        for file_name in file_list:
            #取出一个前景文件
            front_pic = os.path.join(path, file_name)
            # print (front_pic)
            # 2.加载预训练模型
            # 通过加载PaddleHub DeepLabv3+模型(deeplabv3p_xception65_humanseg)实现一键抠图
            module = paddlehub.Module(name="ace2p")
            input_dict = {"image": [front_pic]}
            #print (input_dict)
            #GPU开启下面一行,如果用下面一行，必须安装paddlepaddle-gpu
            results = module.segmentation(data=input_dict,use_gpu = gpu,output_dir=paddlehub_out_path)
            #以下还可以优化吗
            #for result in results: 
            #print(result)


# In[24]:


def cyclegan_cityscapes(fore_image_path,paddlehub_out_path): 
# 1.指定待预测图片路径，遍历循环抠图 保存到 HUMANSEG_SPACE
    g = os.walk(fore_image_path)
    for path,dir_list,file_list in g: 
        file_list = natsorted(file_list)
        for file_name in file_list:
            #取出一个前景文件
            front_pic = [os.path.join(path, file_name)]
            # print (front_pic)
            # 2.加载预训练模型
            # 通过加载PaddleHub DeepLabv3+模型(deeplabv3p_xception65_humanseg)实现一键抠图
            module = paddlehub.Module(name="cyclegan_cityscapes")
            input_dict = {"image": front_pic}
            #print (input_dict)
            #GPU开启下面一行,如果用下面一行，必须安装paddlepaddle-gpu
            results = module.generate(data=input_dict,use_gpu = gpu,batch_size = 256,output_dir=paddlehub_out_path)
            #以下还可以优化吗
            #for result in results: 
            #print(result)


# In[25]:


def mask(fore_image_path,paddlehub_out_path): 
# 1.指定待预测图片路径,遍历循环抠图 保存到 HUMANSEG_SPACE
    g = os.walk(fore_image_path)
    for path,dir_list,file_list in g: 
        file_list = natsorted(file_list)
        for file_name in file_list:
            #取出一个前景文件
            front_pic = [os.path.join(path, file_name)]
            # print (front_pic)
            # 2.加载预训练模型
            # 通过加载PaddleHub 模型(pyramidbox_lite_server_mask)实现口罩检测
            module = paddlehub.Module(name="pyramidbox_lite_server_mask")
            # module = paddlehub.Module(name="pyramidbox_lite_mobile_mask")
            input_dict = {"image": front_pic}
            #print (input_dict)
            #GPU开启下面一行,如果用下面一行,必须安装paddlepaddle-gpu
            results = module.face_detection(data=input_dict,use_gpu = gpu,batch_size = 10,output_dir=paddlehub_out_path)
            #不需要输出则屏蔽下面两行
            for result in results: 
                print(result)


# ###  当然,这里还埋了一些坑,需要自己动手改一改,想动手试试吗,fork即可.
# ### 有什么问题可以 点 Fork记录 旁边的讨论.
# ### 时间不早了凌晨3点半了,白天告诉大家有哪些问题要注意.
# ### mixture_video.py 里面还有一些图像旋转,文件复制,移动的代码,也可以参考. 
# ### 测试的视频和上面演示图不同,是专用来测试滴,也可以下载了看看.
# 

# ## 四、更多帮助
# 

# 飞桨官方技术交流QQ群：703252161
# 
# [PaddleHub issues](http://github.com/PaddlePaddle/PaddleHub/issues)
# 
# [PaddleHub官网地址](http://www.paddlepaddle.org.cn/hub)
# 
# 比赛答疑群二维码：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/3830d5bd9003495899e58dc32f092c273d718eaeff0645f898aba2cecfb73ba6)
# 
