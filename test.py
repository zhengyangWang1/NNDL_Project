import torch
import torch.nn as nn
import torchinfo
from model import CNNTransformerModel


if __name__ == '__main__':
    cts = CNNTransformerModel(vocab_size=128,
                              embed_size=64,
                              num_head=8, )
    image = torch.rand(32, 3, 224, 224)
    text = torch.ones(32, 30).to(torch.int)
    torchinfo.summary(cts, input_data=(image, text))
    pass