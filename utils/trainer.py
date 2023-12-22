# 存放模型的训练函数，日志记录器，路径存放
import os
import time
import json
import logging
import torch
import torch.nn as nn

from ..model import CNNTransformerModel
from .config import Config


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


def cts_train(train_dataloader, config_path=None, ):
    # 设定保存路径变量
    time_str = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join('checkpoints', time_str + 'CNNTransformer')
    model_path = 'model.pth'
    config_path = 'config.json' if config_path == None else config_path # 如果没有指定就使用著文件目录的config.json
    # 设定运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 创建保存路径
    os.makedirs(save_dir, exist_ok=True)
    # 日志记录
    logging.basicConfig(filename='train.log', filemode="w",
                        format="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    logging.info('开始训练')
    # 读取配置
    config = Config()
    config.read_config(config_path)  # 读取参数，打印参数
    # 模型创建
    model = CNNTransformerModel(vocab_size=config.vocab_size,
                                embed_size=config.embed_size,
                                num_head=config.num_head,
                                num_encoder_layer=config.num_decoder,
                                num_decoder_layer=config.num_decoder, ).to(device)
    logging.info('模型创建完成')
    # TODO 损失函数和优化器
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 模型训练
    for epoch in range(config.num_epoch):
        num_samples = 0
        running_loss = 0.0
        for i, (imgs, caps, caplens) in enumerate(train_dataloader):
            # 清空优化器梯度
            optimizer.zero_grad()
            # 设备转移
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            # 处理数据为5*Batchsize，扩展batch
            # forward
            result = model(imgs, caps)
            # TODO 计算损失
            loss = criterion(result)
            # 累计损失
            # 反向传播
            loss.backward()
            optimizer.step()

        average_loss = running_loss / num_samples
        # 日志记录训练信息
        if (epoch + 1) % 1 == 0:
            log_string = f'Epoch: {epoch + 1}, Training Loss: {average_loss:.4f}'
            print(log_string)
            logging.info(log_string)

    # 绘图

    # 保存模型参数，
    torch.save(model.state_dict(), os.path.join(save_dir, model_path))
    # 模型结构，

    # 配置，
    config.save_config(os.path.join(save_dir,'config.json'))
    # 日志
    logging.info('模型训练完成')