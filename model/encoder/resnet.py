from torchvision import models
import torch.nn as nn
import torchinfo


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        # 预训练的resnet101模型
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        # 输出维度 batchsize 2048 7 7 网格表示    # * 解包：当*出现在可迭代对象前面时，它会将可迭代对象拆分为单独的元素。
        self.grid_representation_extractor = nn.Sequential(*(list(resnet.children())[:-2]))

        # 设置参数张量是否需要微调 被nn.Parameter包装的tensor似乎默认True
        for param in self.grid_representation_extractor.parameters():
            param.requires_grad = True

    def forward(self, images):
        """
        :param images: 输入图片 形状  batchsize 3 224 224
        :return:
        """
        output = self.grid_representation_extractor(images)
        return output


if __name__ == '__main__':
    # 查看resnet101的模型结构
    m = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    m = nn.Sequential(*(list(m.children())[:-2]))
    for param in m.parameters():
        # print(param)
        pass
    # print(m)
    # summary总结
    s = torchinfo.summary(m, input_size=(1, 3, 224, 224), device='cpu')
