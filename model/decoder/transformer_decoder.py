import torch
import torch.nn as nn
import torchinfo


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=8, ):
        super(TransformerDecoder, self).__init__()
        # word embedding 词汇embedding
        # 假设输入为 batchsize, seq_length（句子长度），输出为(batchsize,seq_length,8)
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # transformer解码器 输入输出都为(batchsize,seq_length,8)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)

        # 编码onehot向量 对应输出(batchsize,seq_length,vocabsize)
        self.fc = nn.Linear(embed_size, vocab_size)
        # self.softmax=nn.Softmax(dim=2)

    def forward(self, img_encoded, text):
        """

        :param img_encoded: (batchsize,2048,512)
        :param text: (batchsize,seq_length) 可变化
        :return: (batchsize,seq_length,vocab_size) 输出拟合onehot向量计算cross entropy损失
        """
        text_embedding = self.embedding(text)
        decoded = self.transformer_decoder(text_embedding, img_encoded)
        output = self.fc(decoded)
        return output


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
    model = TransformerDecoder(vocab_size=128, embed_size=8)
    # model = TransformerDecoder(vocab_size=128,embed_size=512)
    # img_encoded = torch.rand(20, 2048, 512)
    img_encoded = torch.rand(20, 32, 512)
    text = torch.ones(20, 32).to(torch.int)
    torchinfo.summary(model, input_data=(img_encoded, text))

