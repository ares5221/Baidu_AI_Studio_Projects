#!/usr/bin/env python
# coding: utf-8

# # **一、数据准备**

# In[1]:


'''
解压数据集
'''
get_ipython().system('unzip -q -o data/data1917/train_new.zip')
get_ipython().system('unzip -q -o data/data1917/test_new.zip')


# In[2]:


'''
加载相关类库
'''
import zipfile
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import matplotlib.image as mping

import json
import numpy as np
import cv2
import sys
import time
import h5py
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
from matplotlib import cm as CM
from paddle.utils.plot import Ploter
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[3]:


'''
查看train.json相关信息，重点关注annotations中的标注信息
'''
f = open('/home/aistudio/data/data1917/train.json',encoding='utf-8')
content = json.load(f)

'''
将上面的到的content中的name中的“stage1/”去掉
'''
for j in range(len(content['annotations'])):
    content['annotations'][j]['name'] = content['annotations'][j]['name'].lstrip('stage1').lstrip('/')


# In[4]:



'''
使用高斯滤波变换生成密度图
'''
def gaussian_filter_density(gt):
   
    # 初始化密度图
    density = np.zeros(gt.shape, dtype=np.float32)
    
    # 获取gt中不为0的元素的个数
    gt_count = np.count_nonzero(gt)
    
    # 如果gt全为0，就返回全0的密度图
    if gt_count == 0:
        return density
    
    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            # sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            sigma = 25
        else:
            sigma = np.average(np.array(gt.shape))/2./2. 
        
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density
     


# In[5]:


'''
图片操作：对图片进行resize、归一化，将方框标注变为点标注
返回：resize后的图片 和 gt
'''
def picture_opt(img,ann):
    size_x,size_y = img.size
    train_img_size = (640,480)
    img = img.resize(train_img_size,Image.ANTIALIAS)
    img = np.array(img)                  
    img = img / 255.0

    gt = []
    for b_l in range(len(ann)):
        # 假设人体是使用方框标注的，通过求均值的方法将框变为点
        if 'w' in ann[b_l].keys(): 
            x = (ann[b_l]['x']+(ann[b_l]['x']+ann[b_l]['w']))/2
            y = ann[b_l]['y']+20
            x = (x*640/size_x)/8
            y = (y*480/size_y)/8
            gt.append((x,y))   
        else:
            x = ann[b_l]['x']
            y = ann[b_l]['y']
            x = (x*640/size_x)/8
            y = (y*480/size_y)/8
            gt.append((x,y)) 
   
    return img,gt


# In[6]:


'''
密度图处理
'''
def ground(img,gt):
    imgs = img
    x = imgs.shape[0]/8
    y = imgs.shape[1]/8
    k = np.zeros((int(x),int(y)))

    for i in range(0,len(gt)):
        if int(gt[i][1]) < int(x) and int(gt[i][0]) < int(y):
            k[int(gt[i][1]),int(gt[i][0])]=1

    k = gaussian_filter_density(k)
    return k
    
    


# In[7]:


'''
定义数据生成器
'''
def train_set():
    def inner():
        for ig_index in range(2000):                                                 #遍历所有图片
            if len(content['annotations'][ig_index]['annotation']) == 2:continue
            if len(content['annotations'][ig_index]['annotation']) == 3:continue
            if content['annotations'][ig_index]['ignore_region']:                      #把忽略区域都用像素为0填上
                ig_list = []                                                           #存放忽略区1的数据
                ig_list1 = []                                                          #存放忽略区2的数据
                # print(content['annotations'][ig_index]['ignore_region'])
                if len(content['annotations'][ig_index]['ignore_region'])==1:           #因为每张图的忽略区域最多2个，这里是为1的情况
                    # print('ig1',ig_index)
                    ign_rge = content['annotations'][ig_index]['ignore_region'][0]       #取第一个忽略区的数据
                    for ig_len in range(len(ign_rge)):                                   #遍历忽略区坐标个数，组成多少变型
                        ig_list.append([ign_rge[ig_len]['x'],ign_rge[ig_len]['y']])       #取出每个坐标的x,y然后组成一个小列表放到ig_list
                    ig_cv_img = cv2.imread(content['annotations'][ig_index]['name'])      #用cv2读取一张图片
                    pts = np.array(ig_list,np.int32)                                      #把ig_list转成numpy.ndarray数据格式，为了填充需要
                    cv2.fillPoly(ig_cv_img,[pts],(0,0,0),cv2.LINE_AA)                     #使用cv2.fillPoly方法对有忽略区的图片用像素为0填充
                
                    ig_img = Image.fromarray(cv2.cvtColor(ig_cv_img,cv2.COLOR_BGR2RGB))   #cv2转PIL
                    
                    ann = content['annotations'][ig_index]['annotation']          #把所有标注的信息读取出来
                                                                  
                    ig_im,gt = picture_opt(ig_img,ann)
                    k = ground(ig_im,gt)
                   
                    groundtruth = np.asarray(k)
                    groundtruth = groundtruth.T.astype('float32')
                    ig_im = ig_im.transpose().astype('float32')
                    yield ig_im,groundtruth
                    
                if len(content['annotations'][ig_index]['ignore_region'])==2:           #有2个忽略区域
                    # print('ig2',ig_index)
                    ign_rge = content['annotations'][ig_index]['ignore_region'][0]
                    ign_rge1 = content['annotations'][ig_index]['ignore_region'][1]
                    for ig_len in range(len(ign_rge)):
                        ig_list.append([ign_rge[ig_len]['x'],ign_rge[ig_len]['y']])
                    for ig_len1 in range(len(ign_rge1)):
                        ig_list1.append([ign_rge1[ig_len1]['x'],ign_rge1[ig_len1]['y']])  
                    ig_cv_img2 = cv2.imread(content['annotations'][ig_index]['name'])
                    pts = np.array(ig_list,np.int32)
                    pts1 = np.array(ig_list1,np.int32)
                    cv2.fillPoly(ig_cv_img2,[pts],(0,0,0),cv2.LINE_AA)                
                    cv2.fillPoly(ig_cv_img2,[pts1],(0,0,0),cv2.LINE_AA)
                    
                    ig_img2 = Image.fromarray(cv2.cvtColor(ig_cv_img2,cv2.COLOR_BGR2RGB))   #cv2转PIL
                    
                    ann = content['annotations'][ig_index]['annotation']                    #把所有标注的信息读取出来
                                                                  
                    ig_im,gt = picture_opt(ig_img2,ann)
                    k = ground(ig_im,gt)
                    k = np.zeros((int(ig_im.shape[0]/8),int(ig_im.shape[1]/8)))
                    
                    groundtruth = np.asarray(k)
                    groundtruth = groundtruth.T.astype('float32')
                    ig_im = ig_im.transpose().astype('float32')
                    yield ig_im,groundtruth
                    
            else:
                img = Image.open(content['annotations'][ig_index]['name'])
                ann = content['annotations'][ig_index]['annotation']          #把所有标注的信息读取出来
                
                im,gt = picture_opt(img,ann)
                k = ground(im,gt)
                
                groundtruth = np.asarray(k)
                groundtruth = groundtruth.T.astype('float32')
                im = im.transpose().astype('float32')
                yield im,groundtruth
    return inner



# In[8]:


BATCH_SIZE= 3    #每次取3张
# 设置训练reader
train_reader = paddle.batch(
    paddle.reader.shuffle(
        train_set(), buf_size=512),
    batch_size=BATCH_SIZE)



# # 二、网络配置

# In[9]:


class CNN(fluid.dygraph.Layer):
    '''
    网络
    '''
    def __init__(self):
        super(CNN, self).__init__()

        
        self.conv01_1 = fluid.dygraph.Conv2D(num_channels=3, num_filters=64,filter_size=3,padding=1,act="relu")
        self.pool01=fluid.dygraph.Pool2D(pool_size=2,pool_type='max',pool_stride=2)

        self.conv02_1 = fluid.dygraph.Conv2D(num_channels=64, num_filters=128,filter_size=3, padding=1,act="relu")
        self.pool02=fluid.dygraph.Pool2D(pool_size=2,pool_type='max',pool_stride=2)

        self.conv03_1 = fluid.dygraph.Conv2D(num_channels=128, num_filters=256,filter_size=3, padding=1,act="relu")
        self.pool03=fluid.dygraph.Pool2D(pool_size=2,pool_type='max',pool_stride=2)

        self.conv04_1 = fluid.dygraph.Conv2D(num_channels=256, num_filters=512,filter_size=3, padding=1,act="relu")

        self.conv05_1 = fluid.dygraph.Conv2D(num_channels=512, num_filters=512,filter_size=3,padding=1, act="relu")
      

        self.conv06 = fluid.dygraph.Conv2D(num_channels=512,num_filters=256,filter_size=3,padding=1,act='relu')
        self.conv07 = fluid.dygraph.Conv2D(num_channels=256,num_filters=128,filter_size=3,padding=1,act='relu')
        self.conv08 = fluid.dygraph.Conv2D(num_channels=128,num_filters=64,filter_size=3,padding=1,act='relu')
        self.conv09 = fluid.dygraph.Conv2D(num_channels=64,num_filters=1,filter_size=1,padding=0,act=None)
        

    def forward(self, inputs, label=None):
        """前向计算"""
        out = self.conv01_1(inputs)
        
        out = self.pool01(out)

        out = self.conv02_1(out)
       
        out = self.pool02(out)

        out = self.conv03_1(out)
        
        out = self.pool03(out)

        out = self.conv04_1(out)    

        out = self.conv05_1(out)
        out = self.conv06(out)
        out = self.conv07(out)
        out = self.conv08(out)
        out = self.conv09(out)
        
        return out


# # 三、模型训练 && 四、模型评估

# In[10]:


'''
模型训练
'''
with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):
# with fluid.dygraph.guard(place = fluid.CPUPlace()):
    cnn = CNN()
    optimizer=fluid.optimizer.AdamOptimizer(learning_rate=0.001,parameter_list=cnn.parameters()) 
    for epoch_num in range(5):
        for batch_id, data in enumerate(train_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')           
            y_data = np.array([x[1] for x in data]).astype('float32') 
            y_data = y_data[:,np.newaxis] 
           
            #将Numpy转换为DyGraph接收的输入
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True

            out = cnn(img,label)
            loss = fluid.layers.square_error_cost(out, label)
            avg_loss = fluid.layers.mean(loss)

            #使用backward()方法可以执行反向网络
            avg_loss.backward()
            optimizer.minimize(avg_loss)
             
            #将参数梯度清零以保证下一轮训练的正确性
            cnn.clear_gradients()
            
            dy_param_value = {}
            for param in cnn.parameters():
                dy_param_value[param.name] = param.numpy
                
            if batch_id % 10 == 0:
                print("Loss at epoch {} step {}: {}".format(epoch_num, batch_id, avg_loss.numpy()))
    #保存模型参数
    fluid.save_dygraph(cnn.state_dict(), "cnn")   
    print("Final loss: {}".format(avg_loss.numpy()))


# In[ ]:





# # 五、模型预测

# In[11]:



data_dict = {}

'''
模型预测
'''
with fluid.dygraph.guard():
    model, _ = fluid.dygraph.load_dygraph("cnn")
    cnn = CNN()
    cnn.load_dict(model)
    cnn.eval()

    #获取预测图片列表
    test_zfile = zipfile.ZipFile("/home/aistudio/data/data1917/test_new.zip")
    l_test = []
    for test_fname in test_zfile.namelist()[1:]:
        
        l_test.append(test_fname)
   

    for  index in range(len(l_test)):
       
        test_img = Image.open(l_test[index])
        test_img = test_img.resize((640,480))
        test_im = np.array(test_img)
        test_im = test_im / 255.0
        test_im = test_im.transpose().reshape(3,640,480).astype('float32')
        l_test[index] = l_test[index].lstrip('test').lstrip('/')

        dy_x_data = np.array(test_im).astype('float32')
        dy_x_data=dy_x_data[np.newaxis,:, : ,:]
        img = fluid.dygraph.to_variable(dy_x_data)
        out = cnn(img)
        temp=out[0][0]
        temp=temp.numpy()
        people =np.sum(temp)
        data_dict[l_test[index]]=int(people)
        
import csv

with open('results.csv', 'w') as csvfile:

    fieldnames = ['id', 'predicted']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for k,v in data_dict.items():

        writer.writerow({'id': k, 'predicted':v})
print("结束")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




