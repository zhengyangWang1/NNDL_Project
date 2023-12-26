import torch
import torch.nn as nn
import torchinfo

from .encoder import TransformerEncoder, ResNetEncoder
from .decoder import TransformerDecoder, GRUDecoder
from utils.gpu_mem_track import MemTracker
from utils.data_loader import gen_text_mask
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

    def beam_search(self, images, beam_k, max_len, vocab_size, vocab):
        image_codes = self.encoder(images)
        texts = []
        device = images.device
        # 对每个图像样本执行束搜索
        for image_code in image_codes:
            # 将图像表示复制k份
            image_code = image_code.unsqueeze(0).repeat(beam_k, 1, 1, 1)  # (5, 2048, 7, 7)
            # 生成k个候选句子，初始时，仅包含开始符号<start>
            cur_sents = torch.full((beam_k, 1), vocab['<start>'], dtype=torch.long).to(device)
            cur_sent_embed = self.decoder.cap_embed(cur_sents)[:, 0, :]  # k, cap_dim
            sent_lens = torch.LongTensor([1] * beam_k).to(device)
            # 获得GRU的初始隐状态
            image_code, cur_sent_embed, _, _, hidden_state = \
                self.decoder.init_hidden_state(image_code, cur_sent_embed, sent_lens)

            image_code = image_code.reshape(beam_k, -1)
            image_code = self.decoder.img_embed(image_code)
            # 存储已生成完整的句子（以句子结束符<end>结尾的句子）
            end_sents = []
            # 存储已生成完整的句子的概率
            end_probs = []
            # 存储未完整生成的句子的概率
            probs = torch.zeros(beam_k, 1).to(device)
            k = beam_k
            while True:
                preds, hidden_state = self.decoder.forward_step(image_code[:k], cur_sent_embed,
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
                end_indices = [idx for idx, word in enumerate(word_indices) if word == vocab['<end>']]
                if len(end_indices) > 0:
                    end_probs.extend(values[end_indices])
                    end_sents.extend(cur_sents[end_indices].tolist())
                    # 如果所有的句子都包含结束符，则停止生成
                    k -= len(end_indices)
                    if k == 0:
                        break
                # 查找还需要继续生成词的句子
                cur_indices = [idx for idx, word in enumerate(word_indices)
                               if word != vocab['<end>']]
                if len(cur_indices) > 0:
                    cur_sent_indices = sent_indices[cur_indices]
                    cur_word_indices = word_indices[cur_indices]
                    # 仅保留还需要继续生成的句子、句子概率、隐状态、词嵌入
                    cur_sents = cur_sents[cur_indices]
                    probs = values[cur_indices].view(-1, 1)
                    hidden_state = hidden_state[:, cur_sent_indices, :]
                    cur_sent_embed = self.decoder.cap_embed(
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

    def forward(self, image, text, text_key_padding_mask=None, text_mask=None):
        """
        :param text_mask:
        :param text_key_padding_mask:
        :param image: B*3*224*224 torch浮点张量
        :param text: B*seq_length torch整型张量
        :return:
        """
        # B*3*224*224 -> B*512*embed_size
        img_encoded = self.encoder(image)
        # gpu_tracker.track()
        # B*512*embed_size,(B*seq_length->B*seq_length*embed_size) -> B*seq_length*vocab_size 词的onehot向量
        return self.decoder(img_encoded, text, text_key_padding_mask=text_key_padding_mask, text_mask=text_mask)

    def greedy_search(self, images, max_len, vocab):
        # 贪婪搜索
        vocab_size = len(vocab)
        # 获取图像编码
        image_codes = self.encoder(images)  # (batchsize, 512, 64)
        batch_size = images.size(0)
        device = images.device
        # batchwise的greedy search
        # 初始化输入句子：等batchsize的输入文本，以start开始
        sentences_in = torch.full((batch_size, 1), vocab['<start>'], dtype=torch.long).to(device)
        # bool张量用于给出哪一条句子已经生成完成（True），无需关注后续信息，填充pad
        end_mask = torch.zeros(batch_size, dtype=torch.bool)
        for i in range(max_len):  # 最长次循环次数
            # 前向传播 imagecode (B,img_code_dim,embedsize) sentences_in (B,seqlength) ->
            # out (B,seqlength,vocab_size) -> (B,1,vocab_size) 最后一列token
            pred_next_word = self.decoder(image_codes, sentences_in)[:, -1, :]
            # 贪婪策略只选择一个句子 不需要考虑概率
            # 选出对应词汇 -> (B,1,) 词汇张量 返回整型张量 不需要换类型
            pred_next_word = torch.argmax(pred_next_word, dim=2)
            # 先填充padding 形状 -> (B,1,) 不变
            pred_next_word[end_mask] = vocab['<pad>']
            # 再拼接到sentences_in
            sentences_in = torch.cat((sentences_in, pred_next_word), dim=1)
            # 检查是否有句子新结束 end标识符
            end_bool = pred_next_word.squeeze() == vocab['<end>']
            # 最后修改mask，避免将之前的end填充为padding
            end_mask = end_mask | end_bool  # 或操作，保留为True的部分
        # 转换为list (B,maxlen) ->list of list
        sentences_in = sentences_in.to_list()
        sentences = []
        # 将pad去除
        for sen in sentences_in:
            sen_filtered = [x for x in sen if x != vocab['<pad>']]
            sentences.append(sen_filtered)
        return sentences

    def beam_search(self, images, beam_k, max_len, vocab):
        """
        返回list batchsize条结果list
        :param images:
        :param beam_k:
        :param max_len:
        :param vocab:
        :return:
        """
        vocab_size = len(vocab)
        image_codes = self.encoder(images)  # (batchsize, 512, 64)
        texts = []
        device = images.device
        # 对每个图像样本执行束搜索
        for image_code in image_codes:  # (512, 64)
            # 将图像表示复制k份 对应k束
            image_code = image_code.unsqueeze(0).repeat(beam_k, 1, 1)  # (beamk, image_code_dim, embed_size)(5,512,64)
            # 生成k个候选句子，初始时，仅包含开始符号<start>
            cur_sents = torch.full((beam_k, 1), vocab['<start>'], dtype=torch.long).to(
                device)  # (beamk, seq_length, embed_size)
            # 存储已生成完整的句子（以句子结束符<end>结尾的句子）
            end_sents = []
            # 存储已生成完整的句子的概率
            end_probs = []
            # 存储未完整生成的句子的概率
            probs = torch.zeros(beam_k, 1).to(device)  # (5, 1)
            k = beam_k  # k在找到一个句子之后减1
            while True:
                # 第一次输入imagecode  (k,512,embedsize) text (k,seq_length) -> (k,seq_length, vocab_size)
                preds = self.decoder(image_code[:k], cur_sents)[:, -1, :]  # TODO:fix 应该是-1 获取序列最后一个结果
                #
                # print(preds.shape)
                preds = nn.functional.log_softmax(preds, dim=1)  # TODO 是否应该改成softmax
                # 对每个候选句子采样概率值最大的前k个单词生成k个新的候选句子，并计算概率
                # -> (k, vocab_size)
                # print('=====')
                # print(probs.shape,preds.shape)
                probs = probs.repeat(1, preds.size(1)) + preds  # (5,seqlength, vocab_size) probs是preds的累加
                if cur_sents.size(1) == 1:
                    # 第一步时，所有句子都只包含开始标识符，因此，仅利用其中一个句子计算topk
                    # 返回值和索引 两个张量
                    # print('-----')
                    # print(probs.shape,probs[0].shape,)
                    values, indices = probs[0].topk(k, dim=0, largest=True, sorted=True)
                    # print(values.shape, indices.shape)
                else:
                    # probs: (k, vocab_size) 是二维张量
                    # topk函数直接应用于二维张量会按照指定维度取最大值，这里需要在全局取最大值
                    # 因此，将probs转换为一维张量（reshape），再使用topk函数获取最大的k个值
                    values, indices = probs.view(-1).topk(k, dim=0, largest=True, sorted=True)
                    # print(values.shape,indices.shape)
                # 计算最大的k个值对应的句子索引和词索引
                # 因为是正常展开，使用整除维度就可以获得对应的topk词典索引
                # print(indices.shape,vocab_size)
                sent_indices = torch.div(indices, vocab_size, rounding_mode='trunc')  # 整除，得到对应的句子
                word_indices = indices % vocab_size  # 取模，得到对应的word索引 5
                # 将词拼接在前一轮的句子后，获得此轮的句子
                # print(f'sent_indices{sent_indices.shape}')
                # print(cur_sents[sent_indices].shape, word_indices.unsqueeze(1).shape) # 5,109,1  5,1,109
                cur_sents = torch.cat([cur_sents[sent_indices], word_indices.unsqueeze(1)], dim=1)  # (5, x)
                # 查找此轮生成句子结束符<end>的句子的 索引
                end_indices = [idx for idx, word in enumerate(word_indices) if word == vocab['<end>']]  # 储存结束句子的索引
                if len(end_indices) > 0:
                    # 结束句子的概率添加到结束句子概率列表中，values[end_indices]获得了end的概率 一个列表
                    end_probs.extend(values[end_indices])  # FIXME 为什么不需要tolist 一维张量可以直接转换为单个张量
                    # 存储已经结束的句子
                    end_sents.extend(cur_sents[end_indices].tolist())  # 结束句子
                    # 已经生成句子 搜索k减少
                    k -= len(end_indices)
                    # 如果所有的句子都包含结束符，则停止生成
                    if k == 0:
                        break
                # 查找还需要继续生成词的句子
                cur_indices = [idx for idx, word in enumerate(word_indices)
                               if word != vocab['<end>']]
                if len(cur_indices) > 0:
                    # 仅保留还需要继续生成的句子、句子概率
                    cur_sents = cur_sents[cur_indices]
                    probs = values[cur_indices].view(-1, 1)
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


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO 文本生成器类，实现 贪婪搜索 和 beam搜索


if __name__ == '__main__':
    pass
