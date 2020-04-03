#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
# get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
# get_ipython().system('ls /home/aistudio/work')


# In[7]:


# get_ipython().system('rm -rf __MACOSX')
# get_ipython().system('unzip -q /home/aistudio/data/data23617/characterData.zip')


# In[2]:


#导入需要的包

import numpy as np
import paddle as paddle
import paddle.fluid as fluid
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from multiprocessing import cpu_count
from paddle.fluid.dygraph import Pool2D,Conv2D
# from paddle.fluid.dygraph import FC
# from paddle.fluid.dygraph import Linear


# In[3]:


# 生成车牌字符图像列表
data_path = './data'
character_folders = os.listdir(data_path)
label = 0
LABEL_temp = {}
if(os.path.exists('./train_data.list')):
    os.remove('./train_data.list')
if(os.path.exists('./test_data.list')):
    os.remove('./test_data.list')
for character_folder in character_folders:
    with open('./train_data.list', 'a') as f_train:
        with open('./test_data.list', 'a') as f_test:
            if character_folder == '.DS_Store' or character_folder == '.ipynb_checkpoints' or character_folder == 'data23617':
                continue
            print(character_folder + " " + str(label))
            LABEL_temp[str(label)] = character_folder #存储一下标签的对应关系
            character_imgs = os.listdir(os.path.join(data_path, character_folder))
            for i in range(len(character_imgs)):
                if i%10 == 0: 
                    f_test.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
                else:
                    f_train.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
    label = label + 1
print('图像列表已生成')


# In[4]:


# 用上一步生成的图像列表定义车牌字符训练集和测试集的reader
def data_mapper(sample):
    img, label = sample
    img = paddle.dataset.image.load_image(file=img, is_color=False)
    img = img.flatten().astype('float32') / 255.0
    return img, label
def data_reader(data_list_path):
    def reader():
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)
    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)


# In[5]:


# 用于训练的数据提供器
train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=data_reader('./train_data.list'), buf_size=512), batch_size=128)
# 用于测试的数据提供器
test_reader = paddle.batch(reader=data_reader('./test_data.list'), batch_size=128)


# In[6]:


#定义网络
class MyLeNet(fluid.dygraph.Layer):
    def __init__(self):
        super(MyLeNet,self).__init__()
        self.hidden1_1 = Conv2D(num_channels=1,num_filters=64,filter_size=3,stride=2,padding=1,act='leaky_relu')
        self.hidden1_2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2)
        self.hidden2_1 = Conv2D(num_channels=64,num_filters=128,filter_size=3,stride=1,padding=1,act='leaky_relu')
        self.hidden2_2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2)
        self.hidden3 = Conv2D(num_channels=128,num_filters=256,filter_size=3,stride=1,padding=1,act='leaky_relu')
        self.hidden4 = paddle.fluid.dygraph.Linear(input_dim=1024,output_dim=512,act='leaky_relu')
        self.hidden5 = paddle.fluid.dygraph.Linear(input_dim=512,output_dim=65,act='softmax')
    def forward(self,input):
        x = self.hidden1_1(input)
        x = fluid.layers.dropout(x,dropout_prob=0.2)
        x= self.hidden1_2(x)
        x = self.hidden2_1(x)
        x = fluid.layers.dropout(x,dropout_prob=0.5)
        x = self.hidden2_2(x)
        x = self.hidden3(x)
        x = fluid.layers.reshape(x,shape=[x.shape[0],-1])
        x = fluid.layers.dropout(x,dropout_prob=0.2)
        x = self.hidden4(x)
        y = self.hidden5(x)
        return y


# In[ ]:


with fluid.dygraph.guard():
    model=MyLeNet() #模型实例化
    model.train() #训练模式
    opt=fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
    epochs_num= 300#迭代次数为2
    
    for pass_num in range(epochs_num):
        
        for batch_id,data in enumerate(train_reader()):
            images=np.array([x[0].reshape(1,20,20) for x in data],np.float32)
            labels = np.array([x[1] for x in data]).astype('int64')
            labels = labels[:, np.newaxis]
            image=fluid.dygraph.to_variable(images)
            label=fluid.dygraph.to_variable(labels)
            
            predict=model(image)#预测
            
            loss=fluid.layers.cross_entropy(predict,label)
            avg_loss=fluid.layers.mean(loss)#获取loss值
            
            acc=fluid.layers.accuracy(predict,label)#计算精度
            
            if batch_id!=0 and batch_id%50==0:
                print("train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num,batch_id,avg_loss.numpy(),acc.numpy()))
            
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()            
            
    fluid.save_dygraph(model.state_dict(),'MyLeNet')#保存模型


# In[8]:


#模型校验
with fluid.dygraph.guard():
    accs = []
    model=MyLeNet()#模型实例化
    model_dict,_=fluid.load_dygraph('MyLeNet')
    model.load_dict(model_dict)#加载模型参数
    model.eval()#评估模式
    for batch_id,data in enumerate(test_reader()):#测试集
        images=np.array([x[0].reshape(1,20,20) for x in data],np.float32)
        labels = np.array([x[1] for x in data]).astype('int64')
        labels = labels[:, np.newaxis]
            
        image=fluid.dygraph.to_variable(images)
        label=fluid.dygraph.to_variable(labels)
            
        predict=model(image)#预测
        acc=fluid.layers.accuracy(predict,label)
        accs.append(acc.numpy()[0])
        avg_acc = np.mean(accs)
    print(avg_acc)


# In[9]:


# 对车牌图片进行处理，分割出车牌中的每一个字符并保存
license_plate = cv2.imread('./车牌.png')
gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_RGB2GRAY)
ret, binary_plate = cv2.threshold(gray_plate, 175, 255, cv2.THRESH_BINARY)
result = []
for col in range(binary_plate.shape[1]):
    result.append(0)
    for row in range(binary_plate.shape[0]):
        result[col] = result[col] + binary_plate[row][col]/255
character_dict = {}
num = 0
i = 0
while i < len(result):
    if result[i] == 0:
        i += 1
    else:
        index = i + 1
        while result[index] != 0:
            index += 1
        character_dict[num] = [i, index-1]
        num += 1
        i = index

for i in range(8):
    if i==2:
        continue
    padding = (170 - (character_dict[i][1] - character_dict[i][0])) / 2
    ndarray = np.pad(binary_plate[:,character_dict[i][0]:character_dict[i][1]], ((0,0), (int(padding), int(padding))), 'constant', constant_values=(0,0))
    ndarray = cv2.resize(ndarray, (20,20))
    cv2.imwrite('./' + str(i) + '.png', ndarray)
    
def load_image(path):
    img = paddle.dataset.image.load_image(file=path, is_color=False)
    img = img.astype('float32')
    img = img[np.newaxis, ] / 255.0
    return img


# In[10]:


#将标签进行转换
print('Label:',LABEL_temp)
match = {'A':'A','B':'B','C':'C','D':'D','E':'E','F':'F','G':'G','H':'H','I':'I','J':'J','K':'K','L':'L','M':'M','N':'N',
        'O':'O','P':'P','Q':'Q','R':'R','S':'S','T':'T','U':'U','V':'V','W':'W','X':'X','Y':'Y','Z':'Z',
        'yun':'云','cuan':'川','hei':'黑','zhe':'浙','ning':'宁','jin':'津','gan':'赣','hu':'沪','liao':'辽','jl':'吉','qing':'青','zang':'藏',
        'e1':'鄂','meng':'蒙','gan1':'甘','qiong':'琼','shan':'陕','min':'闽','su':'苏','xin':'新','wan':'皖','jing':'京','xiang':'湘','gui':'贵',
        'yu1':'渝','yu':'豫','ji':'冀','yue':'粤','gui1':'桂','sx':'晋','lu':'鲁',
        '0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9'}
L = 0
LABEL ={}

for V in LABEL_temp.values():
    LABEL[str(L)] = match[V]
    L += 1
print(LABEL)


# In[11]:


#构建预测动态图过程
with fluid.dygraph.guard():
    model=MyLeNet()#模型实例化
    model_dict,_=fluid.load_dygraph('MyLeNet')
    model.load_dict(model_dict)#加载模型参数
    model.eval()#评估模式
    lab=[]
    for i in range(8):
        if i==2:
            continue
        infer_imgs = []
        infer_imgs.append(load_image('./' + str(i) + '.png'))
        infer_imgs = np.array(infer_imgs)
        infer_imgs = fluid.dygraph.to_variable(infer_imgs)
        result=model(infer_imgs)
        lab.append(np.argmax(result.numpy()))
# print(lab)


display(Image.open('./车牌.png'))
print('\n车牌识别结果为：',end='')
for i in range(len(lab)):
    print(LABEL[str(lab[i])],end='')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
