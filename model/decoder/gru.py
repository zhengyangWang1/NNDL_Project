import torch.nn as nn
import torch
import numpy as np
import torchinfo
from torch.nn.utils.rnn import pack_padded_sequence
from model.encoder.resnet import ResNetEncoder
from model.model import Model
from utils.data_loader import dataloader
import torch.optim as optim


class GRUDecoder(nn.Module):
    def __init__(self, img_dim, cap_dim, vocab_size, hidden_size, num_layers, dropout=0.5):
        super(GRUDecoder, self).__init__()
        self.cap_embed = nn.Embedding(vocab_size, cap_dim)
        self.img_embed = nn.Linear(2048*7*7, img_dim)
        self.init_state = nn.Linear(img_dim, num_layers * hidden_size)
        self.gru = nn.GRU(img_dim+cap_dim, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.img_embed.weight.data.uniform_(-0.1, 0.1)
        self.cap_embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, image_code, captions, cap_lens):
        """

        """
        batch_size, image_code_dim = image_code.size(0), image_code.size(1)
        image_code = image_code.permute(0, 2, 3, 1)
        image_code = image_code.view(batch_size, -1, image_code_dim)
        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
        captions = captions[sorted_cap_indices]
        image_code = image_code[sorted_cap_indices]
        hidden_state = self.init_state(image_code.mean(axis=1))
        hidden_state = hidden_state.view(
            batch_size,
            self.gru.num_layers,
            self.gru.hidden_size).permute(1, 0, 2)
        return image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state

    def forward(self, img, captions, cap_lens):
        image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state \
            = self.init_hidden_state(img, captions, cap_lens)
        batch_size = image_code.size(0)

        # 输入序列长度减1，因为最后一个时刻不需要预测下一个词
        lengths = sorted_cap_lens.cpu().numpy() - 1
        predictions = torch.zeros(batch_size, lengths[0], self.fc.out_features)

        img = image_code.reshape(image_code.size(0), -1)
        img_embeds = self.img_embed(img)
        cap_embeds = self.cap_embed(captions)

        for step in range(lengths[0]):
            real_batch_size = np.where(lengths > step)[0].shape[0]

            img = img_embeds[:real_batch_size]
            cap = cap_embeds[:real_batch_size, step, :]
            hidden_stat = hidden_state[:, :real_batch_size, :].contiguous()

            x = torch.cat((img, cap), dim=-1).unsqueeze(0)
            output, _ = self.gru(x, hidden_stat)
            pred = self.fc(self.dropout(output.squeeze(0)))

            predictions[:real_batch_size, step, :] = pred

        return predictions, captions, lengths, sorted_cap_indices


class PackedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PackedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, lengths):
        """
        参数：
            predictions：按文本长度排序过的预测结果
            targets：按文本长度排序过的文本描述
            lengths：文本长度
        """
        predictions = pack_padded_sequence(predictions, lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
        return self.loss_fn(predictions, targets)


if __name__ == '__main__':
    encoder = ResNetEncoder()
    decoder = GRUDecoder(2048, 512, 300, 512, num_layers=1)
    model = Model(encoder, decoder)
    train_data, test_data = dataloader('data/deepfashion-mini', 8, workers=0)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = PackedCrossEntropyLoss()

    # 迭代训练
    for epoch in range(5):  # num_epochs 为训练轮数
        running_loss = 0.0
        for i, (imgs, caps, caplens) in enumerate(train_data):
            # 获取输入数据
            imgs = encoder(imgs)
            predictions, sorted_captions, lengths, sorted_cap_indices = decoder(imgs, caps, caplens)

            loss = loss_fn(predictions, sorted_captions[:, 1:], lengths)

            loss.backward()

            optimizer.step()
