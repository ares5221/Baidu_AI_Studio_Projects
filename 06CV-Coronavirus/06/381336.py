#!/usr/bin/env python
# coding: utf-8

# # 说明
# 本次作业为二种类型，分别是选择题和实践题。其中选择题是不定项选择。实践题需要在提示的地方填上代码，跑通项目。
# 
# ## 资料
# 做作业时可以参考以下资料。
# 
# PaddleSlim代码地址： [https://github.com/PaddlePaddle/PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)
# 
# 文档地址：[https://paddlepaddle.github.io/PaddleSlim/](https://paddlepaddle.github.io/PaddleSlim/)

# # 选择题
# 【1】定点化量化的优点有哪些？
# 
# A. 存带宽  B. 低功耗  C. 低计算资源  D. 低存储体积
# 
# 【2】在常规蒸馏任务中，以下说法正确的是：
# 
# A. 只有teacher model的参数需要更新
# 
# B. 只有student model的参数需要更新
# 
# C. teacher model和student model 的参数都需要更新
# 
# D.teacher model和student model 的参数都不需要更新
# 
# 
# 【3】是否能用MobileNetv1蒸馏ResNet50？
# 
# A: 能
# 
# B: 不能
# 
# 【4】下面方法哪些可以减少模型推理时间？
# 
# A. 只对权重weight进行量化
# 
# B. 对ResNet50模型进行蒸馏提高精度
# 
# C. 对模型进行裁剪，减少通道数
# 
# D. 对权重weight和激活进行量化，预测采用INT8计算
# 
# 
# 【5】NAS的三个关键要素是：
# 
# A. 搜索空间
# 
# B. 搜索算法
# 
# C. 模型优化
# 
# D. 模型评估
# 
# 

# # 选择题答题卡
# 
# 请将每道选择题的答案写在这里：
# 
# 【1】ABCD
# 
# 【2】B
# 
# 【3】A
# 
# 【4】ABCD
# 
# 【5】ABD
# 
# 
# 
# 

# #  图像分类模型量化教程
# 
# 该教程以图像分类模型MobileNetV1为例，说明如何快速使用[量化训练接口](https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-aware)。
# 该示例包含以下步骤：
# 
# 1. 导入依赖
# 2. 构建模型
# 3. 定义输入数据
# 4. 训练模型
# 5. 量化模型 ``这个步骤中需要添加代码``
# 6. 训练和测试量化后的模型
# 
# 以下章节依次介绍每个步骤的内容。
# 
# ## 0. 安装paddleslim
# 
# 

# In[2]:


get_ipython().system('pip install paddleslim')


# ## 1. 导入依赖
# 
# PaddleSlim依赖Paddle1.7版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:
# 
# 

# In[3]:


import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
import logging


# In[4]:


logging.basicConfig(
    level = logging.DEBUG,
    filename = 'test.log'
)
logging.debug('debug message')
logging.info('info message')
logging.warn('warn message')
logging.error('error message')
logging.critical('critical message')


# 
# ## 2. 构建模型
# 
# 该章节构造一个用于对MNIST数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[1, 28, 28]`，输出类别数为10。
# 为了方便展示示例，我们在`paddleslim.models`下预定义了用于构建分类模型的方法，执行以下代码构建分类模型：
# 
# 

# In[5]:


use_gpu = fluid.is_compiled_with_cuda()
exe, train_program, val_program, inputs, outputs = slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=use_gpu)
place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()


# 
# 
# ## 3 定义输入数据
# 
# 为了快速执行该示例，我们选取简单的MNIST数据，Paddle框架的`paddle.dataset.mnist`包定义了MNIST数据的下载和读取。
# 代码如下：
# 
# 

# In[6]:



import paddle.dataset.mnist as reader
train_reader = paddle.batch(
        reader.train(), batch_size=128, drop_last=True)
test_reader = paddle.batch(
        reader.test(), batch_size=128, drop_last=True)
data_feeder = fluid.DataFeeder(inputs, place)


# 
# 
# ## 4. 训练和测试
# 
# 先定义训练和测试函数，正常训练和量化训练时只需要调用函数即可。在训练函数中执行了一个epoch的训练，因为MNIST数据集数据较少，一个epoch就可将top1精度训练到95%以上。
# 

# In[7]:


def train(prog):
    iter = 0
    for data in train_reader():
        acc1, acc5, loss = exe.run(prog, feed=data_feeder.feed(data), fetch_list=outputs)
        if iter % 100 == 0:
            print('train iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        iter += 1
        
def test(prog):
    iter = 0
    res = [[], []]
    for data in test_reader():
        acc1, acc5, loss = exe.run(prog, feed=data_feeder.feed(data), fetch_list=outputs)
        if iter % 100 == 0:
            print('test iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        res[0].append(acc1.mean())
        res[1].append(acc5.mean())
        iter += 1
    print('final test result top1={}, top5={}'.format(np.array(res[0]).mean(), np.array(res[1]).mean()))


# 
# 调用train函数训练分类网络，train_program是在第2步：构建网络中定义的

# In[8]:


train(train_program)


# 
# 调用test函数测试分类网络，val_program是在第2步：构建网络中定义的。

# In[9]:


test(val_program)


# 
# 
# ## 5. 量化模型
# 
# 按照配置在train_program和val_program中加入量化和反量化op.
# 
# 

# In[10]:


place = exe.place
quant_program =  slim.quant.quant_aware(train_program, exe.place, for_test=False)       #请在次数添加你的代码
val_quant_program = slim.quant.quant_aware(val_program, exe.place, for_test=True)    #请在次数添加你的代码


# ## 6 训练和测试量化后的模型¶
# 微调量化后的模型，训练一个epoch后测试。

# In[11]:



train(quant_program)


# 测试量化后的模型，和3.2 训练和测试中得到的测试结果相比，精度相近，达到了无损量化。
# 

# In[12]:


test(val_quant_program)


# In[13]:


float_prog, int8_prog = slim.quant.convert(val_quant_program, exe.place, save_int8=True)
target_vars = [float_prog.global_block().var(name) for name in outputs]
fluid.io.save_inference_model(dirname='./inference_model/float',
        feeded_var_names=[var.name for var in inputs],
        target_vars=target_vars,
        executor=exe,
        main_program=float_prog)
fluid.io.save_inference_model(dirname='./inference_model/int8',
        feeded_var_names=[var.name for var in inputs],
        target_vars=target_vars,
        executor=exe,
        main_program=int8_prog)

