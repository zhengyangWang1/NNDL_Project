from torchvision import models
import torch.nn as nn
import torchinfo

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder,self).__init__()
        # 预训练的resnet101模型
        resnet= models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # self.grid_representation_extractor = nn.Sequential(*(list(resnet.children())[:-2]))
        for param in self.grid_representation_extractor.parameters():
            print()


if __name__ == '__main__':
    # 查看resnet101的模型结构
    m = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    print(m)
    # summary总结
    s=torchinfo.summary(m,input_size=(1,3,224,224),device='cpu')