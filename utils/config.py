import os
import json


class Config:
    def __init__(self):
        # 模型配置类default
        # 模型种类

        self.model_type = 'CNN_GRU'  # 'CNN_GRU' or 'CNN_Transformer'

        # 一般超参数
        self.vocab_size = 109
        self.encoder_lr = 0.0001
        self.decoder_lr = 0.0005
        self.num_epoch = 10
        self.batch_size = 64

        # CNNRNN类
        self.img_dim = 2048
        self.cap_dim = 512
        self.hidden_size = 512
        self.num_layers = 1

        # CNNTTransformer类
        self.embed_size = 32
        self.num_head = 8
        self.dim_ff = 512
        self.num_encoder = 6
        self.num_decoder = 6
        pass

    def read_config(self, json_path):
        # 从json文件读取配置
        with open(json_path, 'r') as file:
            hparam = json.load(file)
        for key, value in hparam.items():
            if hasattr(self, key):
                # 类中有这个超参数的定义
                setattr(self, key, value)
            else:
                print(f'配置{key}未定义，请检查')
        # 检查实现
        assert self.model_type in ['CNN_GRU', 'CNN_Transformer'], "没有这个模型的实现"
        print(hparam)

    def save_config(self, json_path):
        # 保存配置到json文件, 避免保存__name__等属性 好像没有
        hyper_params = vars(self)
        with open(json_path, 'w') as file:
            json.dump(hyper_params, file, indent=4)
        pass


if __name__ == '__main__':
    config = Config()
    config.save_config('config.json')
