import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class PackedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PackedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src, targets, lengths):
        """ 其实也是在词的维度上求得的损失
        :param src: 预测结果
        :param targets: 文本描述
        :param lengths: 文本描述长度列表
        :return:
        """
        src = pack_padded_sequence(src, lengths, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(targets, lengths, batch_first=True, enforce_sorted=False)[0]
        return self.loss_fn(src, targets)


class TokenCrossEntropyLoss(nn.Module):
    def __init__(self, padding_index):
        super(TokenCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_index)

    def forward(self, src, tgt):
        """ 其实也是在词的维度上求得的损失
        :param src: 预测结果 形状 N,S,vocab
        :param tgt: 文本描述 形状 N,S
        :return:
        """
        # src = pack_padded_sequence(src, lengths, batch_first=True, enforce_sorted=False)[0]
        # targets = pack_padded_sequence(targets, lengths, batch_first=True, enforce_sorted=False)[0]
        return self.loss_fn(src.reshape(-1, src.shape[-1]), tgt.reshape(-1))


# FIXME follow https://github.com/ruotianluo/self-critical.pytorch/blob/master/captioning/modules/losses.py
class RewardLoss(nn.Module):
    def __init__(self):
        super(RewardLoss, self).__init__()

    def forward(self, input, seq, reward, reduction='mean'):
        N, L = input.shape[:2]
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq > 0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask

        if reduction == 'none':
            output = output.view(N, L).sum(1) / mask.view(N, L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output
