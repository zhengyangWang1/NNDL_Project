# 存放模型的训练函数，日志记录器，路径存放
import os
import time
import json
import logging
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data_loader import dataloader
from model import CNNRNNStruct
from model import ResNetEncoder, GRUDecoder
from model.loss_function import PackedCrossEntropyLoss
from model.model import CNNTransformerModel
from utils.config import Config
import torch.optim as optim


def train(train_dataloader, test_dataloader, config: Config, ):
    # 设定保存路径变量
    time_str = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join('checkpoints', time_str + config.use_model_type)
    # 设定运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建保存路径
    os.makedirs(save_dir, exist_ok=True)
    # 日志记录
    logging.basicConfig(filename=os.path.join(save_dir, 'train.log'), filemode="w",
                        format="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    logging.info('----------开始训练----------')
    ##############
    # 模型加载或创建
    ##############
    checkpoint = None
    if config.use_model_type == 'CNN_Transformer':
        model = CNNTransformerModel(vocab_size=config.vocab_size,
                                    embed_size=config.CNN_Transformer.embed_size,
                                    num_head=config.CNN_Transformer.num_head,
                                    num_encoder_layer=config.CNN_Transformer.num_decoder,
                                    num_decoder_layer=config.CNN_Transformer.num_decoder,
                                    dim_ff=config.CNN_Transformer.dim_ff, ).to(device)
        logging.info('模型创建完成')
    elif config.use_model_type == 'CNN_GRU':
        if checkpoint is None:  # 模型没有断点
            encoder = ResNetEncoder()
            decoder = GRUDecoder(img_dim=config.CNN_GRU.img_dim,
                                 cap_dim=config.CNN_GRU.cap_dim,
                                 vocab_size=config.vocab_size,
                                 hidden_size=config.CNN_GRU.hidden_size,
                                 num_layers=config.CNN_GRU.num_layers)
            model = CNNRNNStruct(encoder, decoder).to(device)
            logging.info('模型创建完成')
        else:  # 断点部分 TODO 更改加载方式
            checkpoint = torch.load(checkpoint)
            # start_epoch = checkpoint['epoch']
            model = checkpoint
            logging.info('模型加载完成')
    else:
        raise ValueError("model_type not found")

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
        tqdm_param = {
            'total': len(train_dataloader),
            'mininterval': 0.5,
            'dynamic_ncols': True,
        }
        with tqdm(enumerate(train_dataloader), **tqdm_param, desc='Train') as t:
            for i, (imgs, caps, caplens) in t:
                # 清空优化器梯度
                optimizer.zero_grad()
                # 设备转移
                # transfer_start = time.time()
                imgs = imgs.to(device)
                caps = caps.to(device)
                # forward 返回B*seq_length*vocab_size
                # forward_start = time.time()
                if config.use_model_type == 'CNN_Transformer':
                    capsin = caps[:, :-1] # N,seqlength-1 相当于长度减1
                    capsout = caps[:, 1:] # N,seqlength-1 相当于去掉start
                    logits = model(imgs, capsin)
                    capsout = torch.eye(config.vocab_size, device=device)[capsout]  # 在caps所在设备上生成one-hot向量
                    loss = criterion(logits, capsout, caplens)
                elif config.use_model_type == 'CNN_GRU':
                    predictions, sorted_captions, lengths, sorted_cap_indices = model(imgs, caps, caplens)
                    loss = criterion(predictions, sorted_captions[:, 1:], lengths)
                # 累计损失
                num_samples += imgs.size(0)
                running_loss += loss.item()  # 因为是pack_padded之后的张量loss算作是整个batchsize的张量
                l = loss.item()
                # 反向传播
                loss.backward()
                optimizer.step()
                # end = time.time()
                # 进度条设置
                pf = {'loss': f'{l:.4f}', }
                t.set_postfix(pf)
                if i % 30 == 0:  # 日志记录
                    log_string = f'Iter {i:>5}: Loss {l:.4f}'
                    # log_string = f'Iter {i:>5}: Loss {l:.4f}|Transfer data :{forward_start - transfer_start:.2f}s|Forward :{end - forward_start:.2f}s|'
                    logging.info(log_string)

        average_loss = running_loss / num_samples
        # 日志记录训练信息
        if (epoch + 1) % 1 == 0:
            log_string = f'Epoch: {epoch + 1}, Training Loss: {average_loss:.4f}, Time Cost: {time.time() - batch_start:.2f}s'
            print(log_string)
            logging.info(log_string)

        # 每个epoch之后测试模型
        evaluate(test_dataloader, config, model)
        model.train()

        # torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        # 在每个epoch结束时保存
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion': criterion.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_dir, f'model_checkpoint{epoch}.pth'))
        logging.info('模型保存完成')

    # TODO 保存词典

    # 保存模型结构
    with open(os.path.join(save_dir, 'model_structure.txt'), 'w') as f:  # 保存模型层级结构
        f.write(str(model))
    # 保存配置
    config.save_config(os.path.join(save_dir, 'config.json'))
    # 保存日志
    logging.info('----------模型训练完成----------')


# 测试beam search用
def evaluation(test_loader, config: Config, vocab):
    checkpoint = torch.load('checkpoints/12-25_00-45CNN_Transformer/model_checkpoint3.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = ResNetEncoder()
    # decoder = GRUDecoder(img_dim=config.CNN_GRU.img_dim,
    #                      cap_dim=config.CNN_GRU.cap_dim,
    #                      vocab_size=config.vocab_size,
    #                      hidden_size=config.CNN_GRU.hidden_size,
    #                      num_layers=config.CNN_GRU.num_layers)
    # model = CNNRNNStruct(encoder, decoder).to(device)
    model = CNNTransformerModel(vocab_size=config.vocab_size,
                                embed_size=config.CNN_Transformer.embed_size,
                                num_head=config.CNN_Transformer.num_head,
                                num_encoder_layer=config.CNN_Transformer.num_decoder,
                                num_decoder_layer=config.CNN_Transformer.num_decoder,
                                dim_ff=config.CNN_Transformer.dim_ff, ).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        for i, (imgs, caps, caplens) in enumerate(test_loader):
            # 通过束搜索，生成候选文本
            texts = model.beam_search(imgs.to(device), config.beam_k, config.max_len + 2, config.vocab_size, vocab)
            print(texts)
            input("按下回车键继续...")


def evaluate(test_dataloader, config, model=None, model_path=None):
    # 给定模型或者给出加载模型路径，否则报错
    assert (model is None) ^ (model_path is None), '必须指定模型或者给出加载模型路径'
    # 设定运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 日志
    logging.info('----------开始评估----------')
    # 模型正常
    # TODO eval使用路径加载模型
    if config.use_model_type == 'CNN_Transformer':
        pass
    elif config.use_model_type == 'CNN_GRU':
        pass
    model.to(device)
    logging.info('----------评估模型加载完成----------')
    # 加载损失函数
    criterion = PackedCrossEntropyLoss().to(device)
    model.eval()
    with torch.no_grad():
        num_samples = 0
        running_loss = 0.0
        batch_start = time.time()
        tqdm_param = {
            'total': len(test_dataloader),
            'mininterval': 0.5,
            'dynamic_ncols': True,
        }
        with tqdm(enumerate(test_dataloader), **tqdm_param, desc='Eval') as t:
            for i, (imgs, caps, caplens) in t:
                # 设备转移
                imgs = imgs.to(device)
                caps = caps.to(device)
                if config.use_model_type == 'CNN_Transformer':
                    result = model(imgs, caps)
                    caps = torch.eye(config.vocab_size, device=device)[caps]  # 在caps所在设备上生成one-hot向量
                    loss = criterion(result, caps, caplens)
                elif config.use_model_type == 'CNN_GRU':
                    predictions, sorted_captions, lengths, sorted_cap_indices = model(imgs, caps, caplens)
                    loss = criterion(predictions, sorted_captions[:, 1:], lengths)
                # 累计损失
                num_samples += imgs.size(0)
                running_loss += loss.item()  # 因为是pack_padded之后的张量loss算作是整个batchsize的张量

        average_loss = running_loss / num_samples
        log_string = f'Eval, Loss: {average_loss:.4f}, Time Cost: {time.time() - batch_start:.2f}s'
        print(log_string)
        logging.info(log_string)
        logging.info('----------评估完成----------')
