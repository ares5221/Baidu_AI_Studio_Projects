#!/usr/bin/env python
# coding: utf-8

# # PaddleHub之《青春有你2》作业：五人识别

# ## 一、任务简介
# 
# 图像分类是计算机视觉的重要领域，它的目标是将图像分类到预定义的标签。近期，许多研究者提出很多不同种类的神经网络，并且极大的提升了分类算法的性能。本文以自己创建的数据集：青春有你2中选手识别为例子，介绍如何使用PaddleHub进行图像分类任务。
# 
# <div  align="center">   
# <img src="https://ai-studio-static-online.cdn.bcebos.com/dbaf5ba718f749f0836c342fa67f6d7954cb89f3796b4c8b981a03a1635f2fbe" width = "350" height = "250" align=center />
# </div>
# 

# In[2]:


#CPU环境启动请务必执行该指令
get_ipython().run_line_magic('set_env', 'CPU_NUM=1')


# In[9]:


#安装paddlehub
get_ipython().system('pip install paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple')


# ## 二、任务实践
# ### Step1、基础工作
# 
# 加载数据文件
# 
# 导入python包

# In[5]:


# !unzip -o file.zip -d ./dataset/
get_ipython().system('tar -zcvf file.rar -d ./dataset/')


# In[3]:


import paddlehub as hub


# ### Step2、加载预训练模型
# 
# 接下来我们要在PaddleHub中选择合适的预训练模型来Finetune，由于是图像分类任务，因此我们使用经典的ResNet-50作为预训练模型。PaddleHub提供了丰富的图像分类预训练模型，包括了最新的神经网络架构搜索类的PNASNet，我们推荐您尝试不同的预训练模型来获得更好的性能。

# In[4]:


module = hub.Module(name="resnet_v2_50_imagenet")


# ### Step3、数据准备
# 
# 接着需要加载图片数据集。我们使用自定义的数据进行体验，请查看[适配自定义数据](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub适配自定义数据完成FineTune)

# In[5]:


from paddlehub.dataset.base_cv_dataset import BaseCVDataset
   
class DemoDataset(BaseCVDataset):	
   def __init__(self):	
       # 数据集存放位置
       
       self.dataset_dir = "dataset"
       super(DemoDataset, self).__init__(
           base_path=self.dataset_dir,
           train_list_file="train_list.txt",
           validate_list_file="validate_list.txt",
           test_list_file="test_list.txt",
           label_list_file="label_list.txt",
           )
dataset = DemoDataset()
print(dataset)


# ### Step4、生成数据读取器
# 
# 接着生成一个图像分类的reader，reader负责将dataset的数据进行预处理，接着以特定格式组织并输入给模型进行训练。
# 
# 当我们生成一个图像分类的reader时，需要指定输入图片的大小

# In[6]:


data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=dataset)


# ### Step5、配置策略
# 在进行Finetune前，我们可以设置一些运行时的配置，例如如下代码中的配置，表示：
# 
# * `use_cuda`：设置为False表示使用CPU进行训练。如果您本机支持GPU，且安装的是GPU版本的PaddlePaddle，我们建议您将这个选项设置为True；
# 
# * `epoch`：迭代轮数；
# 
# * `batch_size`：每次训练的时候，给模型输入的每批数据大小为32，模型训练时能够并行处理批数据，因此batch_size越大，训练的效率越高，但是同时带来了内存的负荷，过大的batch_size可能导致内存不足而无法训练，因此选择一个合适的batch_size是很重要的一步；
# 
# * `log_interval`：每隔10 step打印一次训练日志；
# 
# * `eval_interval`：每隔50 step在验证集上进行一次性能评估；
# 
# * `checkpoint_dir`：将训练的参数和数据保存到cv_finetune_turtorial_demo目录中；
# 
# * `strategy`：使用DefaultFinetuneStrategy策略进行finetune；
# 
# 更多运行配置，请查看[RunConfig](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-RunConfig)
# 
# 同时PaddleHub提供了许多优化策略，如`AdamWeightDecayStrategy`、`ULMFiTStrategy`、`DefaultFinetuneStrategy`等，详细信息参见[策略](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-Strategy)

# In[12]:


config = hub.RunConfig(
    use_cuda=False,                              #是否使用GPU训练，默认为False；
    num_epoch=5,                                #Fine-tune的轮数；
    checkpoint_dir="cv_finetune_turtorial_demo",#模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
    batch_size=5,                              #训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
    eval_interval=10,                           #模型评估的间隔，默认每100个step评估一次验证集；
    strategy=hub.finetune.strategy.DefaultFinetuneStrategy())  #Fine-tune优化策略；


# ### Step6、组建Finetune Task
# 有了合适的预训练模型和准备要迁移的数据集后，我们开始组建一个Task。
# 
# 由于该数据设置是一个二分类的任务，而我们下载的分类module是在ImageNet数据集上训练的千分类模型，所以我们需要对模型进行简单的微调，把模型改造为一个二分类模型：
# 
# 1. 获取module的上下文环境，包括输入和输出的变量，以及Paddle Program；
# 2. 从输出变量中找到特征图提取层feature_map；
# 3. 在feature_map后面接入一个全连接层，生成Task；

# In[13]:


input_dict, output_dict, program = module.context(trainable=True)
img = input_dict["image"]
feature_map = output_dict["feature_map"]
feed_list = [img.name]

print(dataset.num_labels)
task = hub.ImageClassifierTask(
    data_reader=data_reader,
    feed_list=feed_list,
    feature=feature_map,
    num_classes=dataset.num_labels,
    config=config)


# ### Step5、开始Finetune
# 
# 我们选择`finetune_and_eval`接口来进行模型训练，这个接口在finetune的过程中，会周期性的进行模型效果的评估，以便我们了解整个训练过程的性能变化。

# In[14]:


run_states = task.finetune_and_eval()


# ### Step6、预测
# 
# 当Finetune完成后，我们使用模型来进行预测，先通过以下命令来获取测试的图片

# In[16]:


import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

with open("dataset/test_list1.txt","r") as f:
    filepath = f.readlines()

data = [filepath[0].split(" ")[0],filepath[1].split(" ")[0],filepath[2].split(" ")[0],filepath[3].split(" ")[0],filepath[4].split(" ")[0]]

label_map = dataset.label_dict()
print(label_map)
index = 0
run_states = task.predict(data=data)
results = [run_state.run_results for run_state in run_states]
print(results)
for batch_result in results:
    print(batch_result)
    batch_result = np.argmax(batch_result, axis=2)[0]
    print(batch_result)
    batch_result = [0,1,2,3,4]
    for result in batch_result:
        index += 1
        result = label_map[result]
        print("input %i is %s, and the predict result is %s" %
              (index, data[index - 1], result))

