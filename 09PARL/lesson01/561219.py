#!/usr/bin/env python
# coding: utf-8

# # 强化学习（RL）初印象
# ## Part1 什么是强化学习
# * 强化学习（英语：`Reinforcement learning`，简称`RL`）是机器学习中的一个领域，强调如何基于环境而行动，以取得最大化的预期利益。
# * 核心思想：智能体`agent`在环境`environment`中学习，根据环境的状态`state`（或观测到的`observation`），执行动作`action`，并根据环境的反馈 `reward`（奖励）来指导更好的动作。
# 
# *注意：从环境中获取的状态，有时候叫`state`，有时候叫`observation`，这两个其实一个代表全局状态，一个代表局部观测值，在多智能体环境里会有差别，但我们刚开始学习遇到的环境还没有那么复杂，可以先把这两个概念划上等号。*
# 
# 
# ![强化学习模型](https://ai-studio-static-online.cdn.bcebos.com/992844ad9a8e478b8a4490e7eb6d3a6398b6e2b8fd5c40998a02eef5fceb9fc3)
# 
# ## Part2 强化学习能做什么
# * 游戏（马里奥、Atari、Alpha Go、星际争霸等）
# * 机器人控制（机械臂、机器人、自动驾驶、四轴飞行器等）
# * 用户交互（推荐、广告、NLP等）
# * 交通（拥堵管理等）
# * 资源调度（物流、带宽、功率等）
# * 金融（投资组合、股票买卖等）
# * 其他
# 
# ## Part3 强化学习与监督学习的区别
# * 强化学习、监督学习、非监督学习是机器学习里的三个不同的领域，都跟深度学习有交集。
# * 监督学习寻找输入到输出之间的映射，比如分类和回归问题。
# * 非监督学习主要寻找数据之间的隐藏关系，比如聚类问题。
# * 强化学习则需要在与环境的交互中学习和寻找最佳决策方案。
# * 监督学习处理认知问题，强化学习处理决策问题。
# 
# ## Part4 强化学习的如何解决问题
# * 强化学习通过不断的试错探索，吸取经验和教训，持续不断的优化策略，从环境中拿到更好的反馈。
# * 强化学习有两种学习方案：基于价值(`value-based`)、基于策略(`policy-based`)
# 
# ## Part5 强化学习的算法和环境
# * 经典算法：`Q-learning`、`Sarsa`、`DQN`、`Policy Gradient`、`A3C`、`DDPG`、`PPO`
# * 环境分类：离散控制场景（输出动作可数）、连续控制场景（输出动作值不可数）
# * 强化学习经典环境库`GYM`将环境交互接口规范化为：重置环境`reset()`、交互`step()`、渲染`render()`
# * 强化学习框架库`PARL`将强化学习框架抽象为`Model`、`Algorithm`、`Agent`三层，使得强化学习算法的实现和调试更方便和灵活。

# ## Part6 实践 熟悉强化学习环境
# * `GYM`是强化学习中经典的环境库，下节课我们会用到里面的`CliffWalkingWapper`和`FrozenLake`环境，为了使得环境可视化更有趣一些，直播课视频中演示的Demo对环境的渲染做了封装，感兴趣的同学可以在`PARL`代码库中的`examples/tutorials/lesson1`中下载`gridworld.py`使用。
# 
# * *`PARL`开源库地址：https://github.com/PaddlePaddle/PARL*

# In[1]:


get_ipython().system('pip install gym')


# In[2]:


# -*- coding: utf-8 -*-
import gym
import numpy as np


# In[3]:


if __name__ == '__main__':
    # 环境1：FrozenLake, 可以配置冰面是否是滑的
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up

    # 环境2：CliffWalking, 悬崖环境
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

    # PARL代码库中的`examples/tutorials/lesson1`中`gridworld.py`提供了自定义格子世界的封装，可以自定义配置格子世界的地图

    env.reset()
    for step in range(100):
        action = np.random.randint(0, 4)
        obs, reward, done, info = env.step(action)
        print('step {}: action {}, obs {}, reward {}, done {}, info {}'.format(                step, action, obs, reward, done, info))
        env.render()

