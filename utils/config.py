import os
import json
from dataclasses import dataclass
from pprint import pprint


@dataclass
class ModelConfig:
    pass
    # 模型配置类，自定义需要的模型参数...


class Config:
    def __init__(self):
        # 最多允许二级嵌套，按照自定义的时间顺序排序，
        # 即时调节训练超参数
        self.use_model_type = 'CNN_GRU'  # 'CNN_GRU' or 'CNN_Transformer'
        self.num_epoch = 4
        self.batch_size = 64
        self.eval_batch_size = 512
        self.encoder_lr = 0.0002
        self.decoder_lr = 0.0005

        # 束搜索参数
        self.beam_k = 5  # 束搜索空间

        # 模型特定超参数
        self.CNN_GRU = ModelConfig()
        self.CNN_GRU.img_dim = 2048
        self.CNN_GRU.cap_dim = 512
        self.CNN_GRU.hidden_size = 512
        self.CNN_GRU.num_layers = 1

        self.CNN_Transformer = ModelConfig()
        self.CNN_Transformer.embed_size = 32
        self.CNN_Transformer.num_head = 8
        self.CNN_Transformer.dim_ff = 512
        self.CNN_Transformer.num_encoder = 6
        self.CNN_Transformer.num_decoder = 6

        # 数据集固定超参数
        self.vocab_size = 109
        self.max_len = 25  # 文本最大长度，包含start和end
        self.dataset_path = 'data/deepfashion-mini'
        self.vocab_path = 'data/deepfashion-mini/vocab.json'  # 词典路径

        # 保存路径参数 训练后定义
        self.model_checkpoint_path = None  # 例子''
        self.saved_device_type = None # 保存的模型的device type

    def read_config(self, json_path):
        # 从json文件读取配置
        with open(json_path, 'r') as file:
            hparam = json.load(file)
        for key, value in hparam.items():
            if hasattr(self, key):  # 类中有这个超参数的定义
                if isinstance(value, dict):  # 参数是字典，说明是ModelConfig对象
                    model_config = getattr(self, key)  # 获取ModelConfig对象引用
                    # 参数量检查：未定义的，未赋值的x默认值
                    for k, v in value.items():  # 修改ModelConfig对象属性
                        if hasattr(model_config, k):  # ModelConfig对象有属性k
                            setattr(model_config, k, v)
                        else:
                            print(f'配置{key}.{k}未定义，请检查')
                else:  # 正常参数
                    setattr(self, key, value)
            else:
                print(f'配置{key}未定义，请检查')
        # 检查实现
        assert self.use_model_type in ['CNN_GRU', 'CNN_Transformer'], f"没有{self.use_model_type}模型的实现"
        pprint(hparam)

    def save_config(self, json_path):
        # 保存配置到json文件, 避免保存__name__等属性 好像没有
        hyper_params = vars(self)
        for key, value in hyper_params.items():
            if isinstance(value, ModelConfig):
                hyper_params[key] = vars(value)
        with open(json_path, 'w') as file:
            json.dump(hyper_params, file, indent=4)

    def read_vocab(self):
        # 读取词典，返回词典dict(token映射index)和i2t dict(index映射token，用于还原句子)
        with open(self.vocab_path, 'r') as file:
            vocab = json.load(file)
        i2t = {v: k for k, v in vocab.items()}  # index到token的映射 i2t[0] 可以像列表一样使用
        return vocab, i2t


if __name__ == '__main__':
    config = Config()
    config.read_config('config.json')
    config.save_config('config.json')
    w2i,i2w=config.read_vocab()
    print(w2i,)
    print(i2w)

    # dataclass 创建属性检查
    # model_config = ModelConfig()
    # model_config.a = 1
    # print(model_config.a)
    # model_config.a = 2
    # print(model_config.a)
