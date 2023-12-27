import os
import json
import torch
import random
import signal
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Dataset
from torch.nn import Transformer
from PIL import Image
from torchvision import transforms
from nltk.tokenize import sent_tokenize, word_tokenize

# from nltk.corpus import brown,

"""
# 读取数据，预处理数据
# 图片不需要预处理，直接将图片路径或转化为数组输入模型
# 描述需要预处理，文本描述不是单一句子，需要转化为单一句子，处理成向量

主要函数：
data_preprocess: 进行数据预处理
get_data_loader: 将预处理的数据定义为train_loader和test_loader
"""

import nltk


def data_preprocess(data_file='data/deepfashion-mini', min_word_freq=5, captions_per_image=7, max_len=25):
    """
    :param data_file: 数据集根目录(输入数据：12694张图片，训练集和测试集的json文件(10155, 2538)，json中包含{图片名：描述})
    :param min_word_freq: 构建词汇表时，词汇至少出现的次数，
    :param captions_per_image: 每张图片对应的文本描述数
    :param max_len: 文本包含最大单词数
    :return: none 把处理好的数据存入json文件：train_data.json和test_data.json，同时将词典存为vocab.json
    """
    # nltk word_tokenize需要 下载punkt
    try:
        print(nltk.data.find('tokenizers/punkt'))
    except LookupError:
        print('nltk需要下载punkt，尝试下载')
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(60)
        nltk.download('punkt')
        signal.signal(signal.SIGALRM, signal.SIG_DFL)

    # 读取数据集 数据集格式 字典name.jpg:captions 处理为->
    train_json = os.path.join(data_file, 'train_captions.json')  # 训练集描述路径
    with open(train_json, 'r', encoding='utf-8') as train_file:
        train_data = json.load(train_file)

    test_json = os.path.join(data_file, 'test_captions.json')
    with open(test_json, 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)

    # 将描述转化为列表 dict_values() -> list   py3.7之后的版本中是有顺序的
    train_descriptions = list(train_data.values())

    # 统计词频 统一为小写，并且标点符号算作token
    word_counts = Counter()
    punctuation = [',', '.']
    for text in train_descriptions:
        for char in punctuation:
            text = text.replace(char, f" {char} ")
        words = text.split()
        words = [word.lower() for word in words]
        word_counts.update(words)

    # 过滤最小词频词汇 将符合要求的转换为单词列表
    filtered_words = [word for word, freq in word_counts.items() if freq >= min_word_freq]

    # 建立词汇表 单词:索引 w2i
    vocab = {word: idx + 4 for idx, word in enumerate(filtered_words)}  # 从4开始构建词汇表，0, 1, 2, 3用于特殊标记
    vocab['<pad>'] = 0  # 填充符号
    vocab['<start>'] = 1  # 开始符号
    vocab['<end>'] = 2  # 结束符号
    vocab['<unk>'] = 3  # 未知词汇

    sen_count = Counter()  # test 图像对应描述数量计数器
    len_count = Counter()  # test 描述长度计数器

    def process(data):
        """返回处理好的图像描述列表和文本列表 长度为N和N*captions_per_image"""
        img_paths = []
        sequences = []
        for img, description in data.items():
            # img 图像路径字符串 description 描述文本长字符串
            img_paths.append(os.path.join(data_file, 'images', img))

            # 处理文本
            sentences = sent_tokenize(description)  # 将段落描述分为句子
            sen_count.update([len(sentences)])
            if len(sentences) < captions_per_image:
                # 如果该图片对应的描述数量不足，则补足
                for _ in range(captions_per_image - len(sentences)):
                    sentences.append(random.choice(sentences))
                captions = sentences
            else:
                # 如果该图片对应的描述数量超了，则随机下采样 会导致同一个顺序混乱
                captions = random.sample(sentences, k=captions_per_image)

            # 对文本描述进行编码
            for sentence in captions:
                words = word_tokenize(sentence.lower())  # 转换为小写并分词
                sequence = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in words] + [
                    vocab['<end>']]  # 前面加上start 后面加上end 中间没有查找到的转换为unk
                len_count.update([len(sequence)])
                sequences.append(sequence)
        return img_paths, sequences

    train_img_paths, train_sequences = process(train_data)
    test_img_paths, test_sequences = process(test_data)

    print(len(train_img_paths), len(train_sequences))
    print(len(test_img_paths), len(test_sequences))

    train_data = {'IMAGES': train_img_paths, 'CAPTIONS': train_sequences}
    test_data = {'IMAGES': test_img_paths, 'CAPTIONS': test_sequences}

    # 存储词典 添加了缩进使json更易读
    with open(os.path.join(data_file, 'vocab.json'), 'w') as vocab_file:
        json.dump(vocab, vocab_file, indent=4)
    # 存储数据
    with open(os.path.join(data_file, 'train_data.json'), 'w') as train_file:
        json.dump(train_data, train_file, indent=2)
    with open(os.path.join(data_file, 'test_data.json'), 'w') as test_file:
        json.dump(test_data, test_file, indent=2)

    print(f'词典大小{len(vocab)}')
    print(f'训练数据集 图像{len(train_data["IMAGES"])} 文本{len(train_data["CAPTIONS"])}')
    print(f'测试数据集 图像{len(test_data["IMAGES"])} 文本{len(test_data["CAPTIONS"])}')
    print('---------- Data preprocess successfully! ----------')


class CustomDataset(Dataset):
    def __init__(self, data_path, vocab_path, captions_per_image=7, max_len=25, transform=None):
        """
        :param data_path: 预处理好的数据文件路径
        :param vocab_path: 词典文件路径
        :param captions_per_image: 每张图片对应的文本描述数
        :param max_len: 文本包含最大单词数
        :param transform: 需要进行的图片预处理操作
        """
        # 读取train/test_data.json
        with open(data_path, 'r') as file:
            self.data = json.load(file)
        # 读取词典vocab.json
        with open(vocab_path, 'r') as file:
            self.vocab = json.load(file)
        self.transform = transform
        self.caption_per_image = captions_per_image
        self.max_len = max_len
        self.data_size = len(self.data['CAPTIONS'])  # 定义为caption的描述

    def __len__(self):
        return self.data_size

    def __getitem__(self, i):
        # 读取图像 一个图像对应caption_per_image条句子
        img = Image.open(self.data['IMAGES'][i // self.caption_per_image]).convert('RGB')
        # 图像预处理
        if self.transform is not None:
            img = self.transform(img)

        # 每条文本描述的长度
        caplen = len(self.data['CAPTIONS'][i])

        # 填充caption，对齐长度
        # print(self.data['CAPTIONS'][i])
        # print([self.vocab['<pad>']] * (self.max_len - caplen))
        caption = torch.LongTensor(self.data['CAPTIONS'][i] + [self.vocab['<pad>']] * (self.max_len - caplen))

        # 3,224,224  25  1 ->
        return img, caption, caplen


class EvalDataset(Dataset):
    def __init__(self, data_path, vocab_path, captions_per_image=7, max_len=25, transform=None):
        """
        :param data_path: 预处理好的数据文件路径
        :param vocab_path: 词典文件路径
        :param captions_per_image: 每张图片对应的文本描述数
        :param max_len: 文本包含最大单词数
        :param transform: 需要进行的图片预处理操作
        """
        # 读取train/test_data.json
        with open(data_path, 'r') as file:
            self.data = json.load(file)
        # 读取词典vocab.json
        with open(vocab_path, 'r') as file:
            self.vocab = json.load(file)
        self.transform = transform
        self.caption_per_image = captions_per_image
        self.max_len = max_len
        self.data_size = len(self.data['IMAGES'])  # 定义为caption的描述

    def __len__(self):
        return self.data_size

    def __getitem__(self, i):
        # 读取图像 一个图像对应caption_per_image条句子
        # img = Image.open(self.data['IMAGES'][i // self.caption_per_image]).convert('RGB')
        img = Image.open(self.data['IMAGES'][i]).convert('RGB')
        # 图像预处理
        if self.transform is not None:
            img = self.transform(img)

        caplens = []
        captions = []

        for j in range(self.caption_per_image):
            caplen = len(self.data['CAPTIONS'][i * 7 + j])
            captions.append(
                torch.LongTensor(self.data['CAPTIONS'][i * 7 + j] + [self.vocab['<pad>']] * (self.max_len - caplen)))
            caplens.append(caplen)

        captions = torch.stack(captions)

        # 3,224,224:torch.Float  7,25:torch.LongTensor   7:list
        return img, captions, caplens


def get_dataloader(data_dir, batch_size, eval_batch_size, workers=4):
    """
    :param eval_batch_size:  评估批处理量
    :param data_dir: 数据集根目录
    :param batch_size: 批处理量
    :param workers: 进程数，默认为4
    :return: train_loader, test_loader
    """
    # 在data_process中得到的以下预处理数据
    train_data_dir = os.path.join(data_dir, 'train_data.json')
    test_data_dir = os.path.join(data_dir, 'test_data.json')
    vocab_dir = os.path.join(data_dir, 'vocab.json')

    # 定义图像预处理方法
    # 原始图像大小 750*1101
    transform = transforms.Compose([
        transforms.Pad(padding=(175, 0), padding_mode='edge'),  # 左右分别填充175 变成1100，1101 的大小
        transforms.CenterCrop(1100),  # 从中间裁剪 1100矩形
        transforms.Resize(224),  # 调整图像大小为224x224（适应resnet101预训练的输入，进行了缩放）
        transforms.ToTensor(),  # 将图像转换为张量（Tensor）格式。将图像的每个像素值映射到0到1的范围内，并调整维度顺序为(C,H,W)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # FIXME ? 对图像进行标准化处理，即减去均值并除以标准差。使用给定的均值（[0.485, 0.456, 0.406]）和标准差（[0.229, 0.224, 0.225]）对每个通道的像素进行归一化处理
    ])

    # 创建训练和测试数据集对象
    train_dataset = CustomDataset(train_data_dir, vocab_dir, transform=transform)
    test_dataset = CustomDataset(test_data_dir, vocab_dir, transform=transform)
    eval_dataset = EvalDataset(test_data_dir, vocab_dir, transform=transform)

    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=workers)

    # 测试拆分数据集
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=workers)

    return train_loader, test_loader, eval_loader


def gen_text_mask(text, pad, device):
    # 获取数据mask
    # 输入N,seq_length
    # 输出text_padding_mask: N,seq_length text_mask:N*numhead,seq_length,seq_length
    # text_padding_mask
    text_padding_mask = (text == pad).to(device)
    text_mask = Transformer.generate_square_subsequent_mask(text.size(1)).to(device)
    return text_padding_mask, text_mask


def signal_handler(signum, frame):
    raise TimeoutError('下载超时，请尝试别的方式下载punkt')


if __name__ == '__main__':
    pass
    # 在项目根目录运行
    # data_preprocess()


    # 测试
    train_loader, test_loader, eval_loader = get_dataloader("data/deepfashion-mini", 64, 512, 0)
    tqdm_param = {
        'total': len(eval_loader),
        'mininterval': 0.5,
        # 'miniters': 3,
        # 'unit':'iter',
        'dynamic_ncols': True,
        # 'desc':'Training',
        # 'postfix':'final'
    }
    with tqdm(enumerate(eval_loader), **tqdm_param, desc='Training') as t:
        for i, (imgs, caps, caplens) in t:
            pf = {
                'i': i,
                'loss': 0.12,
                'acc': 0.33
            }
            print(imgs.shape, caps.shape, len(caplens))
            input('waiting ')
            t.set_postfix(pf)

    # ------------------------------------------------------------------
    # # 图片处理测试
    # custom_transform = transforms.Compose([
    #     transforms.Pad(padding=(175, 0), padding_mode='edge'),
    #     transforms.CenterCrop(1100),
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    # ])
    #
    # image = Image.open("data/deepfashion-mini/images/MEN-Denim-id_00000080-01_7_additional.jpg")
    #
    # transformed_image = custom_transform(image)
    #
    # plt.imshow(transformed_image.permute(1, 2, 0))  # 将Tensor转换为numpy数组并显示
    # plt.title('Padded Image')
    #
    # plt.show()
