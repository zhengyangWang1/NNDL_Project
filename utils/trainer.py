# 存放模型的训练函数，日志记录器，路径存放
import os
import time
import json
import logging
import torch
import torch.nn as nn
from utils.data_loader import dataloader
from model import CNNRNNStruct
from model import ResNetEncoder, GRUDecoder
from model.loss_function import PackedCrossEntropyLoss
from model.model import CNNTransformerModel
from utils.config import Config
import torch.optim as optim


def train(train_dataloader, config: Config, ):
    # 设定保存路径变量
    time_str = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join('checkpoints', time_str + config.model_type)
    model_path = 'model.pth'
    config_path = 'config.json'
    # 设定运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建保存路径
    os.makedirs(save_dir, exist_ok=True)
    # 日志记录
    logging.basicConfig(filename=os.path.join(save_dir, 'train.log'), filemode="w",
                        format="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    logging.info('开始训练')
    # 读取配置
    # config = Config()
    # config.read_config(config_path)  # 读取参数，打印参数
    # 模型加载或创建
    checkpoint = None
    if checkpoint is None:
        if config.model_type == 'transformer':
            model = CNNTransformerModel(vocab_size=config.vocab_size,
                                        embed_size=config.embed_size,
                                        num_head=config.num_head,
                                        num_encoder_layer=config.num_decoder,
                                        num_decoder_layer=config.num_decoder,
                                        dim_ff=config.dim_ff, ).to(device)
        else:
            encoder = ResNetEncoder()
            decoder = GRUDecoder(config.img_dim, config.cap_dim, config.vocab_size, config.hidden_size,
                                 config.num_layers)
            model = CNNRNNStruct(encoder, decoder).to(device)
        logging.info('模型创建完成')
    else:
        checkpoint = torch.load(checkpoint)
        # start_epoch = checkpoint['epoch']
        model = checkpoint
        logging.info('模型加载完成')

    # 损失函数和优化器
    criterion = PackedCrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW([{"params": filter(lambda p: p.requires_grad, model.encoder.parameters()),
                                    "lr": config.encoder_lr},
                                   {"params": filter(lambda p: p.requires_grad, model.decoder.parameters()),
                                    "lr": config.decoder_lr}])
    # 模型训练
    model.train()
    for epoch in range(config.num_epoch):
        num_samples = 0
        running_loss = 0.0
        batch_start = time.time()

        for i, (imgs, caps, caplens) in enumerate(train_dataloader):
            # 清空优化器梯度
            optimizer.zero_grad()
            # 设备转移
            start = time.time()
            imgs = imgs.to(device)
            caps = caps.to(device)
            # caplens = caplens.to(device)
            print(f'Transfer data :{time.time() - start:.4f}| ', end='')
            # 处理数据为7*Batchsize，扩展batch
            # forward 返回B*seq_length*vocab_size
            start = time.time()
            if config.model_type == 'transformer':
                result = model(imgs, caps)
                eye_tensor = torch.eye(config.vocab_size, device=device)  # 在caps所在设备上生成one-hot向量
                caps = eye_tensor[caps]
                loss = criterion(result, caps, caplens)
            else:
                predictions, sorted_captions, lengths, sorted_cap_indices = model(imgs, caps, caplens)
                loss = criterion(predictions, sorted_captions[:, 1:], lengths)
            print(f'Forward :{time.time() - start:.4f}| ', end='')

            # 累计损失
            num_samples += imgs.size(0)
            running_loss += imgs.size(0) * loss.item()
            # 反向传播
            print(f'Iter {i},Loss {loss.item():.4f}')
            loss.backward()
            optimizer.step()

        average_loss = running_loss / num_samples
        # 日志记录训练信息
        if (epoch + 1) % 1 == 0:
            log_string = f'Epoch: {epoch + 1}, Training Loss: {average_loss:.4f}, Time Cost: {time.time() - batch_start:.4f}'
            print(log_string)
            logging.info(log_string)

        # 在每个epoch结束时保存
        state = {
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer
        }
        torch.save(state, os.path.join(save_dir, model_path))

    # 模型结构
    with open(os.path.join(save_dir, 'model_structure.txt'), 'w') as f:  # 保存模型层级结构
        f.write(str(model))
    # 配置
    config.save_config(os.path.join(save_dir, 'config.json'))
    # 日志
    logging.info('模型训练完成')

#
# def cnn_gru_train():
#     data_dir = 'data/deepfashion-mini'
#     last_checkpoint = 'checkpoints/last_cnn_gru.ckpt'
#
#     vocab_path = os.path.join(data_dir, 'vocab.json')
#     with open(vocab_path, 'r') as f:
#         vocab = json.load(f)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     start_epoch = 0
#     checkpoint = None
#     if checkpoint is None:
#         # 定义模型
#         encoder = ResNetEncoder()
#         decoder = GRUDecoder(2048, 512, len(vocab), 512, num_layers=1)
#         model = CNNRNNStruct(encoder, decoder)
#     else:
#         checkpoint = torch.load(checkpoint)
#         # start_epoch = checkpoint['epoch']
#         model = checkpoint
#     train_data, test_data = dataloader('data/deepfashion-mini', 32, workers=4)
#
#     # 定义损失函数和优化器
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     loss_fn = PackedCrossEntropyLoss().to(device)
#     model.to(device)
#     model.train()
#     # 迭代训练
#     num_epochs = 10
#     for epoch in range(num_epochs - start_epoch):  # num_epochs 为训练轮数
#         num_sample = 0
#         running_loss = 0.0
#         for i, (imgs, caps, caplens) in enumerate(train_data):
#             # 获取输入数据
#             optimizer.zero_grad()
#             imgs = imgs.to(device)
#             caps = caps.to(device)
#             caplens = caplens.to(device)
#
#             predictions, sorted_captions, lengths, sorted_cap_indices = model(imgs, caps, caplens)
#             loss = loss_fn(predictions, sorted_captions[:, 1:], lengths)
#             num_sample += imgs.shape[0]
#             running_loss += loss.item() * imgs.shape[0]  # 尝试释放累积历史记录：https://pytorch.org/docs/stable/notes/faq.html
#             loss.backward()
#             optimizer.step()
#             if i % 50 == 0:
#                 print('batch: ', i)
#         average_loss = running_loss / num_sample
#         print(f"Epoch [{epoch + start_epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
#
#         state = {
#             'epoch': epoch,
#             'model': model,
#             'optimizer': optimizer
#         }
#         torch.save(state, last_checkpoint)
