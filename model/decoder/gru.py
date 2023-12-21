import torch.nn as nn
import torch
import numpy as np
import torchinfo
from torch.nn.utils.rnn import pack_padded_sequence
from model.encoder.resnet import ResNetEncoder
from model.model import Model
from utils.data_loader import dataloader
import torch.optim as optim


# GRU解码器
class GRUDecoder(nn.Module):
    def __init__(self, img_dim, cap_dim, vocab_size, hidden_size, num_layers=1, dropout=0.5):
        """

        :param img_dim: 图片的输入维度，在resnet101中为2048
        :param cap_dim: 描述的维度
        :param vocab_size: 词表的大小（词表中包含词的数量）
        :param hidden_size: 隐藏状态的大小
        :param num_layers: GRU的层数，默认为1
        :param dropout: dropout操作的参数，默认为0.5
        """
        super(GRUDecoder, self).__init__()
        self.cap_embed = nn.Embedding(vocab_size, cap_dim)  # 将描述进行embedding
        self.img_embed = nn.Linear(2048 * 7 * 7, img_dim)  # 将图片进行embedding，减小维度
        self.init_state = nn.Linear(img_dim, num_layers * hidden_size)
        self.gru = nn.GRU(img_dim + cap_dim, hidden_size, num_layers)  # 定义GRU
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.img_embed.weight.data.uniform_(-0.1, 0.1)
        self.cap_embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, image_code, captions, cap_lens):
        """
        初始化隐藏状态，同时对图片描述进行sort处理
        :param image_code: 图片
        :param captions: 描述
        :param cap_lens: 描述长度
        :return: 处理维度后的图片，排序后的描述，排序后的描述长度，描述索引，隐藏状态
        """
        batch_size, image_code_dim = image_code.size(0), image_code.size(1)
        image_code = image_code.permute(0, 2, 3, 1)  # batch_size*7*7*2048
        image_code = image_code.view(batch_size, -1, image_code_dim)  # batch_size*49*2048
        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
        captions = captions[sorted_cap_indices]
        image_code = image_code[sorted_cap_indices]
        hidden_state = self.init_state(image_code.mean(axis=1))
        hidden_state = hidden_state.view(
            batch_size,
            self.gru.num_layers,
            self.gru.hidden_size).permute(1, 0, 2)  # 1*batch_size*hidden_size
        return image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state

    def forward(self, img, captions, cap_lens):
        # 初始化隐藏状态
        image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state \
            = self.init_hidden_state(img, captions, cap_lens)
        batch_size = image_code.size(0)

        # 输入序列长度减1，因为最后一个时刻不需要预测下一个词
        lengths = sorted_cap_lens.cpu().numpy() - 1
        predictions = torch.zeros(batch_size, lengths[0], self.fc.out_features).to(captions.device)  # batch_size*max_length*out_size

        # 对数据进行embedding操作
        img = image_code.reshape(image_code.size(0), -1)  # (batch_size, 2048*7*7)
        img_embeds = self.img_embed(img)  # (batch_size, img_dim)
        cap_embeds = self.cap_embed(captions)  # (batch_size, cap_dim)

        # 一个时间步处理一个词
        for step in range(lengths[0]):
            # 只取有意义的进行训练
            real_batch_size = np.where(lengths > step)[0].shape[0]
            img = img_embeds[:real_batch_size]
            cap = cap_embeds[:real_batch_size, step, :]
            hidden_stat = hidden_state[:, :real_batch_size, :].contiguous()

            # 使用真实数据的img，cap作为输入，而不是使用模型自己的预测结果(Teacher Forcing模式)
            x = torch.cat((img, cap), dim=-1).unsqueeze(0)  # 在第0维增加时间步维度

            # 前向传播过程
            output, hidden_state = self.gru(x, hidden_stat)
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

    pass

