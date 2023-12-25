import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import math
import gc


class TransformerDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_head,
                 num_decoder_layer=6,
                 dim_ff=512,
                 tracker=None, ):
        super(TransformerDecoder, self).__init__()
        self.num_head = num_head
        # word embedding 词汇embedding
        # 假设输入为 batchsize, seq_length（句子长度），输出为(batchsize,seq_length,embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = PositionalEncoding(embed_size)
        # transformer解码器 输入输出都为(batchsize,seq_length,embed_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size,
                                                        nhead=num_head,
                                                        dim_feedforward=dim_ff,
                                                        batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer,
                                                         num_layers=num_decoder_layer)  # TODO 是否需要norm

        self.fc = nn.Linear(embed_size, vocab_size)
        # self.tracker=tracker
        # self.softmax=nn.Softmax(dim=2)

    def forward(self, img_encoded, text, text_key_padding_mask=None, text_mask=None, ):
        """
        :param text_mask: mask attention的mask
        :param text_key_padding_mask: padding位置的mask
        :param img_encoded: (batchsize,2048,512)
        :param text: (batchsize,seq_length) 可变化
        :return: (batchsize,seq_length,vocab_size) 输出拟合onehot向量计算cross entropy损失
        """
        # if self.tracker is not None:
        #     self.tracker.track()
        text_embedding = self.embedding(text)
        # if self.tracker is not None:
        #     self.tracker.track()
        text_embedding = self.positional_embedding(text_embedding)
        # if self.tracker is not None:
        #     self.tracker.track()
        decoded = self.transformer_decoder(text_embedding,
                                           img_encoded,
                                           tgt_mask=text_mask,
                                           tgt_key_padding_mask=text_key_padding_mask)  # 不关注句子中的padding填充信息矩阵
        # if self.tracker is not None:
        #     self.tracker.track()
        output = self.fc(decoded)

        # if self.tracker is not None:
        #     self.tracker.track()
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.01, max_len=100):
        """
        :param d_model: 输入feature维度，也就是embed_size
        :param dropout:
        :param max_len:
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 对数空间内计算一次位置编码
        pe = torch.zeros(max_len, d_model)  # shape (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # shape(max_len,1)  从0-max_len整数变成float
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))  #
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1,max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 输入 batch_size,seq_length,embed_size + 1,seq_length(切片),embed_size
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


if __name__ == '__main__':
    pass
    # 测试自己的模型

    # model = TransformerDecoder(vocab_size=128, embed_size=512, num_head=8)
    # img_encoded = torch.rand(8, 1024, 512)
    # text = torch.ones(8, 32).to(torch.int)
    # torchinfo.summary(model, input_data=(img_encoded, text))
