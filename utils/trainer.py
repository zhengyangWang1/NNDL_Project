# 存放模型的训练函数，日志记录器，路径存放
import os
import time
import json
import logging

# 保存路径

# 数据集读取

# 按需要可能有多个train，也有可能有模型重构
def train():
    logging.basicConfig(filename='train.log', filemode="w",
                        format="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    logging.info('开始训练')
    # 创建保存路径

    # 读取模型参数config.json

    # 打印参数
    # 模型创建
    # 模型训练

    # 绘图

    # 保存模型参数，模型结构，配置，日志
    pass

