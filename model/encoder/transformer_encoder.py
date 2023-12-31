import torch
import torch.nn as nn
import torchinfo
from torchvision import models
import gc

gc.collect()
torch.cuda.empty_cache()


class TransformerEncoder(nn.Module):
    def __init__(self,
                 grid_embed_size,
                 num_head,
                 num_encoder_layer=6,
                 dim_ff=512,
                 tracker=None):
        super(TransformerEncoder, self).__init__()
        # CNN网格表示生成 采用方案1的resnet101作为网格表示提取器，返回网格表示 (batchsize,2048,7,7)
        # resnet101 = models.resnet101(models.ResNet101_Weights.DEFAULT) # 0.16写法
        resnet101 = models.resnet101(pretrained=True)
        # TODO 修改下载的权重的保存位置 现在为/root/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth
        self.grid_extract = nn.Sequential(*(list(resnet101.children())[:-2]),
                                          nn.Conv2d(2048, 512, kernel_size=1))
        for param in self.grid_extract.parameters():
            param.requires_grad = True  # 需要微调

        # 图像网格embedding
        self.flatten = nn.Flatten(2, 3)
        self.grid_embed = GridEmbedding(grid_embed_size)  # (2048, grid_embed_size)

        # transformer
        # 输入形状 batchsize,seq,feature 比如 batchsize,2048,512
        # d_model 是特征数量 nhead是多头自注意力的头数
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=grid_embed_size,
                                                        nhead=num_head,
                                                        dim_feedforward=dim_ff,
                                                        batch_first=True)
        # 堆叠多层transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layer)
        # self.tracker = tracker

    def forward(self, img, img_mask=None):
        """img_mask No
        :param img: (batchsize,3,224,224)
        :return: (batchsize,2048,embed_size)
        """
        # B*3*224*224 -> B*2048*49
        # if self.tracker is not None:
        #     self.tracker.track()
        grid_representation = self.flatten(self.grid_extract(img))
        # B*2048*7*7 -> B*2048*64
        # 大小 121Mb * batch_size
        # if self.tracker is not None:
        #     self.tracker.track()
        grid_embedding = self.grid_embed(grid_representation)
        # del grid_representation
        # 大小 121Mb * batch_size
        # if self.tracker is not None:
        #     self.tracker.track()
        # B*2048*64 -> B*2048*64
        encoded = self.transformer_encoder(grid_embedding)
        # 大小 121Mb * batch_size
        # if self.tracker is not None:
        #     self.tracker.track()
        return encoded


class GridEmbedding(nn.Module):
    def __init__(self, grid_embed_size=64):
        super(GridEmbedding, self).__init__()
        # 全连接和自适应两种方式 自适应参数少但是不适合针对扩大情况
        self.fc1 = nn.Linear(7 * 7, grid_embed_size)
        # self.relu=nn.ReLU()
        # self.fc2 = nn.Linear(256, 512)
        # self.aap2d = nn.AdaptiveAvgPool2d((14, 14))
        # self.flatten = nn.Flatten(2, 3)

    def forward(self, grid_features):
        """
        任务是输入图像的网格表示并返回图像网格表示的embedding向量
        :return:
        """
        # grid_features=grid_features.reshape(grid_features.shape[0],grid_features.shape[1],-1)
        # fc1out=self.fc1(grid_features)
        # fc1out = self.relu(fc1out)
        # grid_embedding = self.fc2(fc1out)
        # grid_embedding = self.flatten(self.aap2d(grid_features))
        grid_embedding = self.fc1(grid_features)
        return grid_embedding


if __name__ == '__main__':
    encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=512)
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
    src = torch.rand(8, 2048, 32)
    out = encoder_layer(src)
    # torchinfo.summary(encoder_layer, depth=10, input_data=src)

    # 测试编码器
    model = TransformerEncoder(64, 8)
    image = torch.rand(32, 3, 224, 224)
    torchinfo.summary(model, input_data=image)

"""
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
TransformerEncoder                            [32, 512, 64]             83,008
├─Sequential: 1-1                             [32, 512, 7, 7]           --
│    └─Conv2d: 2-1                            [32, 64, 112, 112]        9,408
│    └─BatchNorm2d: 2-2                       [32, 64, 112, 112]        128
│    └─ReLU: 2-3                              [32, 64, 112, 112]        --
│    └─MaxPool2d: 2-4                         [32, 64, 56, 56]          --
│    └─Sequential: 2-5                        [32, 256, 56, 56]         --
│    │    └─Bottleneck: 3-1                   [32, 256, 56, 56]         75,008
│    │    └─Bottleneck: 3-2                   [32, 256, 56, 56]         70,400
│    │    └─Bottleneck: 3-3                   [32, 256, 56, 56]         70,400
│    └─Sequential: 2-6                        [32, 512, 28, 28]         --
│    │    └─Bottleneck: 3-4                   [32, 512, 28, 28]         379,392
│    │    └─Bottleneck: 3-5                   [32, 512, 28, 28]         280,064
│    │    └─Bottleneck: 3-6                   [32, 512, 28, 28]         280,064
│    │    └─Bottleneck: 3-7                   [32, 512, 28, 28]         280,064
│    └─Sequential: 2-7                        [32, 1024, 14, 14]        --
│    │    └─Bottleneck: 3-8                   [32, 1024, 14, 14]        1,512,448
│    │    └─Bottleneck: 3-9                   [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-10                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-11                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-12                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-13                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-14                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-15                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-16                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-17                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-18                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-19                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-20                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-21                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-22                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-23                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-24                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-25                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-26                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-27                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-28                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-29                  [32, 1024, 14, 14]        1,117,184
│    │    └─Bottleneck: 3-30                  [32, 1024, 14, 14]        1,117,184
│    └─Sequential: 2-8                        [32, 2048, 7, 7]          --
│    │    └─Bottleneck: 3-31                  [32, 2048, 7, 7]          6,039,552
│    │    └─Bottleneck: 3-32                  [32, 2048, 7, 7]          4,462,592
│    │    └─Bottleneck: 3-33                  [32, 2048, 7, 7]          4,462,592
│    └─Conv2d: 2-9                            [32, 512, 7, 7]           1,049,088
├─Flatten: 1-2                                [32, 512, 49]             --
├─GridEmbedding: 1-3                          [32, 512, 64]             --
│    └─Linear: 2-10                           [32, 512, 64]             3,200
├─TransformerEncoder: 1-4                     [32, 512, 64]             --
│    └─ModuleList: 2-11                       --                        --
│    │    └─TransformerEncoderLayer: 3-34     [32, 512, 64]             83,008
│    │    └─TransformerEncoderLayer: 3-35     [32, 512, 64]             83,008
│    │    └─TransformerEncoderLayer: 3-36     [32, 512, 64]             83,008
│    │    └─TransformerEncoderLayer: 3-37     [32, 512, 64]             83,008
│    │    └─TransformerEncoderLayer: 3-38     [32, 512, 64]             83,008
│    │    └─TransformerEncoderLayer: 3-39     [32, 512, 64]             83,008
===============================================================================================
Total params: 44,133,504
Trainable params: 44,133,504
Non-trainable params: 0
Total mult-adds (G): 251.23
===============================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 8325.56
Params size (MB): 174.21
Estimated Total Size (MB): 8519.04
===============================================================================================
"""
