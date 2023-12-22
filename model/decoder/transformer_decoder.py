import torch
import torch.nn as nn
import torchinfo


class TransformerDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_head,
                 num_decoder_layer=6, ):
        super(TransformerDecoder, self).__init__()
        # word embedding 词汇embedding
        # 假设输入为 batchsize, seq_length（句子长度），输出为(batchsize,seq_length,embed_size)
        # TODO position_embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # transformer解码器 输入输出都为(batchsize,seq_length,embed_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size,
                                                        nhead=num_head,
                                                        batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer,
                                                         num_layers=num_decoder_layer)

        # FIXME  编码onehot向量 对应输出(batchsize,seq_length,vocabsize) ？
        self.fc = nn.Linear(embed_size, vocab_size)
        # self.softmax=nn.Softmax(dim=2)

    def forward(self, img_encoded, text, imgcode_mask=None, text_mask=None, text_key_padding_mask=None):
        """
        :param img_encoded: (batchsize,2048,512)
        :param text: (batchsize,seq_length) 可变化
        :return: (batchsize,seq_length,vocab_size) 输出拟合onehot向量计算cross entropy损失
        """
        # tgt_mask=None, memory_mask=None
        text_embedding = self.embedding(text)
        decoded = self.transformer_decoder(text_embedding,
                                           img_encoded,
                                           tgt_mask=text_mask, # TODO 只关注之前的信息，应该有函数生成
                                           tgt_key_padding_mask=text_key_padding_mask) # TODO 不关注句子中的padding填充信息矩阵
        output = self.fc(decoded)
        return output


class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()

    def forward(self):
        pass


if __name__ == '__main__':
    # 测试pytorch的transformer层实现 ，输出模型结构和结果
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    memory = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)
    # out = transformer_decoder(tgt, memory)
    # torchinfo.summary(decoder_layer,input_data=(tgt, memory))
    # torchinfo.summary(transformer_decoder, input_data=(tgt, memory))

    # 测试自己的模型
    model = TransformerDecoder(vocab_size=128, embed_size=196)
    # model = TransformerDecoder(vocab_size=128,embed_size=512)
    # img_encoded = torch.rand(20, 2048, 512)
    img_encoded = torch.rand(20, 1024, 196)
    text = torch.ones(20, 32).to(torch.int)
    torchinfo.summary(model, input_data=(img_encoded, text))
