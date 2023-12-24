import os
import json


class Config:
    def __init__(self):
        # 模型配置类default
        # 模型种类
        self.model_type = 'cnn_gnn'  # 'cnn_gnn' or 'transformer'
        self.vocab_size = 109

        # 一般超参数
        self.encoder_lr = 0.0001
        self.decoder_lr = 0.0005
        self.num_epoch = 10
        self.batch_size = 8

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
