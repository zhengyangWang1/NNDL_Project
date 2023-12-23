import torch.optim as optim
import os
import json
import torch

from utils.data_loader import dataloader, data_process
from utils.trainer import cts_train
from utils.config import Config
from model.loss_function import PackedCrossEntropyLoss
from model import ResNetEncoder, GRUDecoder, CNNRNNStruct
from model import CNNTransformerModel

import nltk

# nltk.download('punkt')


if __name__ == '__main__':
    # 数据处理
    # data_process()
    # 加载参数
    config = Config()
    config.read_config('config.json')  # 读取参数，打印参数
    # 数据加载
    train_loader, test_loader = dataloader('data/deepfashion-mini', config.batch_size, workers=0)

    # 模型训练
    cts_train(train_loader, config)

    # data_dir = 'data/deepfashion-mini'
    # last_checkpoint = 'checkpoints/last_cnn_gru.ckpt'
    #
    # vocab_path = os.path.join(data_dir, 'vocab.json')
    # with open(vocab_path, 'r') as f:
    #     vocab = json.load(f)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # start_epoch = 0
    # checkpoint = None
    # if checkpoint is None:
    #     # 定义模型
    #     encoder = ResNetEncoder()
    #     decoder = GRUDecoder(2048, 256, len(vocab), 256, num_layers=1)
    #     model = CNNRNNStruct(encoder, decoder)
    # else:
    #     checkpoint = torch.load(checkpoint)
    #     start_epoch = checkpoint['epoch']
    #     model = checkpoint['model']
    # train_data, test_data = dataloader('data/deepfashion-mini', 32, workers=4)
    #
    # # 定义损失函数和优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = PackedCrossEntropyLoss().to(device)
    # model.to(device)
    # model.train()
    # # 迭代训练
    # num_epochs = 10
    # for epoch in range(num_epochs - start_epoch):  # num_epochs 为训练轮数
    #     num_sample = 0
    #     running_loss = 0.0
    #     for i, (imgs, caps, caplens) in enumerate(train_data):
    #         # 获取输入数据
    #         optimizer.zero_grad()
    #         imgs = imgs.to(device)
    #         caps = caps.to(device)
    #         caplens = caplens.to(device)
    #         #
    #         # grid = encoder(imgs)
    #         # predictions, sorted_captions, lengths, sorted_cap_indices = decoder(grid, caps, caplens)
    #         #
    #         predictions, sorted_captions, lengths, sorted_cap_indices = model(imgs,caps,caplens)
    #         loss = loss_fn(predictions, sorted_captions[:, 1:], lengths)
    #         num_sample += imgs.shape[0]
    #         running_loss += loss * imgs.shape[0]
    #         loss.backward()
    #         optimizer.step()
    #         if i % 50 == 0:
    #             print('batch: ', i)
    #     state = {
    #         'epoch': epoch,
    #         # 'step': i,
    #         'model': model,
    #         'optimizer': optimizer
    #     }
    #     torch.save(model, last_checkpoint)
    #     average_loss = running_loss / num_sample
    #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

