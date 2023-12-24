import torch
import torch.nn as nn
import torchinfo

from .encoder import TransformerEncoder, ResNetEncoder
from .decoder import TransformerDecoder, GRUDecoder
from utils.gpu_mem_track import MemTracker
import numpy as np

# gpu_tracker = MemTracker()


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

    def beam_search(self, images, beam_k, max_len):
        vocab_size = len(self.vocab)
        image_codes = self.encoder(images)
        texts = []
        device = images.device
        # 对每个图像样本执行束搜索
        for image_code in image_codes:
            # 将图像表示复制k份
            image_code = image_code.unsqueeze(0).repeat(beam_k, 1, 1, 1)
            # 生成k个候选句子，初始时，仅包含开始符号<start>
            cur_sents = torch.full((beam_k, 1), self.vocab['<start>'], dtype=torch.long).to(device)
            cur_sent_embed = self.decoder.embed(cur_sents)[:, 0, :]
            sent_lens = torch.LongTensor([1] * beam_k).to(device)
            # 获得GRU的初始隐状态
            image_code, cur_sent_embed, _, _, hidden_state = \
                self.decoder.init_hidden_state(image_code, cur_sent_embed, sent_lens)
            # 存储已生成完整的句子（以句子结束符<end>结尾的句子）
            end_sents = []
            # 存储已生成完整的句子的概率
            end_probs = []
            # 存储未完整生成的句子的概率
            probs = torch.zeros(beam_k, 1).to(device)
            k = beam_k
            while True:
                preds, _, hidden_state = self.decoder.forward_step(image_code[:k], cur_sent_embed,
                                                                   hidden_state.contiguous())
                # -> (k, vocab_size)
                preds = nn.functional.log_softmax(preds, dim=1)
                # 对每个候选句子采样概率值最大的前k个单词生成k个新的候选句子，并计算概率
                # -> (k, vocab_size)
                probs = probs.repeat(1, preds.size(1)) + preds
                if cur_sents.size(1) == 1:
                    # 第一步时，所有句子都只包含开始标识符，因此，仅利用其中一个句子计算topk
                    values, indices = probs[0].topk(k, 0, True, True)
                else:
                    # probs: (k, vocab_size) 是二维张量
                    # topk函数直接应用于二维张量会按照指定维度取最大值，这里需要在全局取最大值
                    # 因此，将probs转换为一维张量，再使用topk函数获取最大的k个值
                    values, indices = probs.view(-1).topk(k, 0, True, True)
                # 计算最大的k个值对应的句子索引和词索引
                sent_indices = torch.div(indices, vocab_size, rounding_mode='trunc')
                word_indices = indices % vocab_size
                # 将词拼接在前一轮的句子后，获得此轮的句子
                cur_sents = torch.cat([cur_sents[sent_indices], word_indices.unsqueeze(1)], dim=1)
                # 查找此轮生成句子结束符<end>的句子
                end_indices = [idx for idx, word in enumerate(word_indices) if word == self.vocab['<end>']]
                if len(end_indices) > 0:
                    end_probs.extend(values[end_indices])
                    end_sents.extend(cur_sents[end_indices].tolist())
                    # 如果所有的句子都包含结束符，则停止生成
                    k -= len(end_indices)
                    if k == 0:
                        break
                # 查找还需要继续生成词的句子
                cur_indices = [idx for idx, word in enumerate(word_indices)
                               if word != self.vocab['<end>']]
                if len(cur_indices) > 0:
                    cur_sent_indices = sent_indices[cur_indices]
                    cur_word_indices = word_indices[cur_indices]
                    # 仅保留还需要继续生成的句子、句子概率、隐状态、词嵌入
                    cur_sents = cur_sents[cur_indices]
                    probs = values[cur_indices].view(-1, 1)
                    hidden_state = hidden_state[:, cur_sent_indices, :]
                    cur_sent_embed = self.decoder.embed(
                        cur_word_indices.view(-1, 1))[:, 0, :]
                # 句子太长，停止生成
                if cur_sents.size(1) >= max_len:
                    break
            if len(end_sents) == 0:
                # 如果没有包含结束符的句子，则选取第一个句子作为生成句子
                gen_sent = cur_sents[0].tolist()
            else:
                # 否则选取包含结束符的句子中概率最大的句子
                gen_sent = end_sents[end_probs.index(max(end_probs))]
            texts.append(gen_sent)
        return texts

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
                                          # tracker=gpu_tracker,
                                          )
        self.decoder = TransformerDecoder(vocab_size,
                                          embed_size,
                                          num_head,
                                          num_decoder_layer,
                                          dim_ff=dim_ff,
                                          # tracker=gpu_tracker,
                                          )

    def forward(self, image, text):
        """
        :param image: B*3*224*224 torch浮点张量
        :param text: B*seq_length torch整型张量
        :return:
        """
        # B*3*224*224 -> B*2048*embed_size
        img_encoded = self.encoder(image)
        # gpu_tracker.track()
        # B*2048*embed_size,(B*seq_length->B*seq_length*embed_size) -> B*seq_length*vocab_size 词的onehot向量
        return self.decoder(img_encoded, text)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO 文本生成器类，实现 贪婪搜索 和 beam搜索



if __name__ == '__main__':
    pass
