# 可能区分为编码器和解码器，也可能有外部调用
# 存放模型的参数
# 示例可修改
import torch
import torch.nn as nn
import torchinfo
from . import TransformerEncoder, TransformerDecoder
from . import ResNetEncoder, GRUDecoder


class CNNRNNStruct(nn.Module):
    def __init__(self, encoder, decoder):
        super(CNNRNNStruct, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        pass

    def forward(self, imgs, caps, caplens):
        grid = self.encoder(imgs)
        predictions, sorted_captions, lengths, sorted_cap_indices = self.decoder(grid, caps, caplens)
        return predictions, sorted_captions, lengths, sorted_cap_indices


class CNNTransformerStruct(nn.Module):
    def __init__(self):
        super(CNNTransformerStruct, self).__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()

    def forward(self):



if __name__ == '__main__':
    pass
