import os
import json
import torch
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

"""
# 读取数据，预处理数据
# 图片不需要预处理，直接将图片路径或转化为数组输入模型
# 描述需要预处理，文本描述不是单一句子，需要转化为单一句子，处理成向量

主要函数：
data_process: 进行数据预处理
data_loader: 将预处理的数据定义为train_loader和test_loader
"""
import nltk
# nltk.download('punkt')

def split_into_sentences(text):
    """
    将段落文本分割成句子
    :param text:
    :return:
    """
    sentences = sent_tokenize(text)
    return sentences


# 单词分词并转换为词汇表中的索引
def text_to_sequence(sentence, vocab):
    words = word_tokenize(sentence.lower())  # 分词并转换为小写
    sequence = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in words] + [vocab['<end>']]
    return sequence


def data_process(data_file='data/deepfashion-mini', min_word_freq=5):
    """
    :param data_file: 数据集根目录(输入数据：12694张图片，训练集和测试集的json文件(10155, 2538)，json中包含{图片名：描述})
    :param min_word_freq: 构建词汇表时，词汇至少出现的次数
    :return: none (把处理好的数据存入json文件：train_data.json和test_data.json)
    """

    # 读取json文件
    train_json = os.path.join(data_file, 'train_captions.json')  # 训练集描述路径
    with open(train_json, 'r', encoding='utf-8') as train_file:
        train_data = json.load(train_file)  # 读取json文件内容

    test_json = os.path.join(data_file, 'test_captions.json')
    with open(test_json, 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)
    # print(test_data)

    # 将描述转化为列表
    train_descriptions = list(train_data.values())
    test_descriptions = list(test_data.values())

    # 将图片路径转化为列表
    image_data = os.path.join(data_file, 'images')  # 图片路径
    train_img_paths = [os.path.join(image_data, f) for f in train_data.keys()]
    test_img_paths = [os.path.join(image_data, f) for f in test_data.keys()]
    # print(test_img_paths)

    # 统计词频
    word_counts = Counter()
    for text in train_descriptions:
        words = text.split()  # 分词
        word_counts.update(words)

    # 过滤词汇
    filtered_words = [word for word, freq in word_counts.items() if freq >= min_word_freq]

    # 建立词汇表
    vocab = {word: idx + 4 for idx, word in enumerate(filtered_words)}  # 从4开始构建词汇表，0, 1, 2, 3用于特殊标记
    vocab['<pad>'] = 0  # 用于填充
    vocab['<start>'] = 1  # 开始符号
    vocab['<end>'] = 2  # 结束符号
    vocab['<unk>'] = 3  # 未知词汇

    # 将图片描述进行编码
    all_train_captions = []
    all_test_captions = []
    for description in train_descriptions:
        sentences = split_into_sentences(description)  # 将段落描述分为句子

        caption_sequence = []
        for sentence in sentences:
            sequence = text_to_sequence(sentence, vocab)  # 转化为编码
            caption_sequence.append(sequence)  # 将一张图片的所有描述句作为一个列表

        all_train_captions.append(caption_sequence)

    for description in test_descriptions:
        sentences = split_into_sentences(description)

        caption_sequence = []
        for sentence in sentences:
            sequence = text_to_sequence(sentence, vocab)
            caption_sequence.append(sequence)

        all_test_captions.append(caption_sequence)

    train_data = {'IMAGES': train_img_paths, 'CAPTIONS': all_train_captions}
    test_data = {'IMAGES': test_img_paths, 'CAPTIONS': all_test_captions}

    # 存储词典
    with open(os.path.join(data_file, 'vocab.json'), 'w') as vocab_file:
        json.dump(vocab, vocab_file)
    # 存储数据
    with open(os.path.join(data_file, 'train_data.json'), 'w') as train_file:
        json.dump(train_data, train_file)
    with open(os.path.join(data_file, 'test_data.json'), 'w') as test_file:
        json.dump(test_data, test_file)

    print('---------- Data preprocess successfully! ----------')


class CustomDataset(Dataset):
    def __init__(self, data_file, transform=None):
        """
        :param data_file: 预处理好的数据文件路径
        :param transform: 需要进行的图片预处理操作
        """
        with open(data_file, 'r') as file:
            self.data = json.load(file)
        self.transform = transform

    def __len__(self):
        return len(self.data['IMAGES'])

    def __getitem__(self, idx):
        img_path = self.data['IMAGES'][idx]
        captions = self.data['CAPTIONS'][idx]

        # 读取图像并进行必要的处理
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # 将文本描述转换为张量形式
        caption = [torch.LongTensor(caption) for caption in captions]

        # 返回图像和对应的多个描述的张量列表
        return image, caption


def dataloader(data_dir, batch_size, workers=4):
    """
    :param data_dir: 数据集根目录
    :param batch_size: 批处理量
    :param workers: 进程数，默认为4
    :return: train_loader, test_loader
    """
    train_data_dir = os.path.join(data_dir, 'train_data.json')
    test_data_dir = os.path.join(data_dir, 'test_data.json')

    # 定义图像预处理方法
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建训练和测试数据集对象
    train_dataset = CustomDataset(train_data_dir, transform=transform)
    test_dataset = CustomDataset(test_data_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    return train_loader, test_loader


if __name__ == '__main__':
    data_process()
    train_loader, test_loader = dataloader('data/deepfashion-mini', 64)

    # 测试
    for batch_data in train_loader:
        inputs, labels = batch_data

    # train_iter= iter(train_loader)
    # batch_data= next(train_iter)
    # image,caption= batch_data
    # print(image,caption)

