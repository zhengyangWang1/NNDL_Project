# 可能区分为编码器和解码器，也可能有外部调用
# 存放模型的参数
# 示例可修改
import torch
import torch.nn as nn
import torchinfo




class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        pass

    def forward(self, x):

        """
        编解码器的大致架构
        :param x:
        :return:
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


if __name__ == '__main__':
    pass
