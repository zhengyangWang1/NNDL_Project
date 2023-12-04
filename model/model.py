# 可能区分为编码器和解码器，也可能有外部调用
# 存放模型的参数
# 示例可修改
import torch
import torch.nn as nn
import torchinfo

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        pass
