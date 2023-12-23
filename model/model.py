import torch
import torch.nn as nn
import torchinfo

from .encoder import TransformerEncoder, ResNetEncoder
from .decoder import TransformerDecoder, GRUDecoder
from utils.gpu_mem_track import MemTracker

gpu_tracker = MemTracker()


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


class CNNTransformerModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size=64,
                 num_head=8,
                 num_encoder_layer=6,
                 num_decoder_layer=6,
                 dim_ff=512):
        """
        :param vocab_size: 文本词典的大小
        :param embed_size: embedding向量维度 必须能够被num_head整除
        :param num_head: 多头注意力 头的数量
        :param num_encoder_layer: transformer编码器层数量
        :param num_decoder_layer: transformer解码器层数量
        """
        super(CNNTransformerModel, self).__init__()
        assert embed_size % num_head == 0, "embedding_size不能被num_head整除"
        self.encoder = TransformerEncoder(embed_size,
                                          num_head,
                                          num_encoder_layer,
                                          dim_ff=dim_ff,
                                          tracker=gpu_tracker, )
        self.decoder = TransformerDecoder(vocab_size,
                                          embed_size,
                                          num_head,
                                          num_decoder_layer,
                                          dim_ff=dim_ff,
                                          tracker=gpu_tracker, )

    def forward(self, image, text):
        """
        :param image: B*3*224*224 torch浮点张量
        :param text: B*seq_length torch整型张量
        :return:
        """
        # B*3*224*224 -> B*2048*embed_size
        img_encoded = self.encoder(image)
        gpu_tracker.track()
        # B*2048*embed_size,(B*seq_length->B*seq_length*embed_size) -> B*seq_length*vocab_size 词的onehot向量
        return self.decoder(img_encoded, text)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # TODO 文本生成器类，实现 贪婪搜索 和 beam搜索


if __name__ == '__main__':
    pass
