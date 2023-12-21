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

    def forward(self, imgs, caps, caplens):
        """
        编解码器的大致架构
        :param x:
        :return:
        """
        grid = self.encoder(imgs)
        predictions, sorted_captions, lengths, sorted_cap_indices = self.decoder(grid, caps, caplens)
        return predictions, sorted_captions, lengths, sorted_cap_indices


if __name__ == '__main__':
    pass
