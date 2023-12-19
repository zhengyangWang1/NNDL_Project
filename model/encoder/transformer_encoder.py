import torch
import torch.nn as nn
import torchinfo

from resnet import ResNetEncoder


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        # CNN网格表示生成 采用方案1的resnet101作为网格表示提取器，返回网格表示 (batchsize,2048,7,7)
        self.grid_extract = GridRepresentationExtractor()

        # 图像网格embedding
        # self.embed = nn.Embedding(vocab_size, embed_size) 词向量的embedding方法
        # 全连接实现或者
        self.grid_embed = GridEmbedding()
        # positional encoding 不需要

        # transformer
        # 输入形状 batchsize,seq,feature 比如 batchsize,2048,512
        # d_model 是特征数量 nhead是多头自注意力的头数
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        # 堆叠多层transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        # repeat

    def forward(self, image):
        """
        :param image: (batchsize,3,224,224)
        :return: (batchsize,2048,512)
        """
        grid_representation = self.grid_extract(image)
        grid_embedding = self.grid_embed(grid_representation)
        encoded = self.transformer_encoder(grid_embedding)
        return encoded


class GridEmbedding(nn.Module):
    def __init__(self):
        super(GridEmbedding, self).__init__()
        self.fc1=nn.Linear(49,256)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        pass

    def forward(self,grid_features):
        """
        任务是输入图像的网格表示并返回图像网格表示的embedding向量
        batchsize,2048,7,7 -> batchsize,2048,512
        :return:
        """
        grid_features=grid_features.reshape(grid_features.shape[0],grid_features.shape[1],-1)
        # 型转变化为 batchsize，2048，49
        fc1out=self.fc1(grid_features)
        fc1out = self.relu(fc1out)
        grid_embedding = self.fc2(fc1out)
        return grid_embedding


class GridRepresentationExtractor(nn.Module):
    def __init__(self):
        super(GridRepresentationExtractor, self).__init__()
        self.resnet_encoder = ResNetEncoder()
        pass

    def forward(self, images):
        """
        任务是使用预训练模型提取输入图像数据的网格特征
        编码为512维度
        :return:
        """
        encoded_grid_features = self.resnet_encoder(images)
        return encoded_grid_features


if __name__ == '__main__':
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    '''
    d_model （int）- 输入中预期特征的数量（必填）。
    nhead （int）- 多头注意力模型中的头数（必填）。
    dim_feedforward (int) - 前馈网络模型的维度（默认值=2048）。
    dropout （float）-- dropout 值（默认值=0.1）。
    activation（Union[str, Callable[[张量],张量]]）- 中间层的激活函数，可以是字符串（"relu "或 "gelu"）或一元可调用函数。默认：relu
    layer_norm_eps（float）--层归一化分量中的 eps 值（默认值=1e-5）。
    batch_first （bool）- 如果为 True，则以 (batch, seq, feature) 的形式提供输入和输出张量。默认值：False (seq, batch, feature)。
    norm_first (bool) - 若为 True，则在进行注意和前馈操作前分别进行层规范处理。否则将在之后进行。默认值：False (after).
    bias (bool) - 如果设置为 False，线性层和 LayerNorm 层将不会学习加法偏置。默认值：默认为 True。
    '''
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    src = torch.rand(10, 32, 512)
    # out = encoder_layer(src)
    # torchinfo.summary(encoder_layer, input_data=src)

    # 测试编码器
    model= TransformerEncoder()
    image = torch.rand(32, 3, 224,224)
    torchinfo.summary(model, input_data=image)

"""
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
TransformerEncoder                                      [32, 2048, 512]           3,152,384
├─GridRepresentationExtractor: 1-1                      [32, 2048, 7, 7]          --
│    └─ResNetEncoder: 2-1                               [32, 2048, 7, 7]          --
│    │    └─Sequential: 3-1                             [32, 2048, 7, 7]          42,500,160
├─GridEmbedding: 1-2                                    [32, 2048, 512]           --
│    └─Linear: 2-2                                      [32, 2048, 256]           12,800
│    └─ReLU: 2-3                                        [32, 2048, 256]           --
│    └─Linear: 2-4                                      [32, 2048, 512]           131,584
├─TransformerEncoder: 1-3                               [32, 2048, 512]           --
│    └─ModuleList: 2-5                                  --                        --
│    │    └─TransformerEncoderLayer: 3-2                [32, 2048, 512]           3,152,384
│    │    └─TransformerEncoderLayer: 3-3                [32, 2048, 512]           3,152,384
│    │    └─TransformerEncoderLayer: 3-4                [32, 2048, 512]           3,152,384
│    │    └─TransformerEncoderLayer: 3-5                [32, 2048, 512]           3,152,384
│    │    └─TransformerEncoderLayer: 3-6                [32, 2048, 512]           3,152,384
│    │    └─TransformerEncoderLayer: 3-7                [32, 2048, 512]           3,152,384
=========================================================================================================
Total params: 64,711,232
Trainable params: 64,711,232
Non-trainable params: 0
Total mult-adds (G): 249.59
=========================================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 8713.40
Params size (MB): 170.58
Estimated Total Size (MB): 8903.25
=========================================================================================================
"""