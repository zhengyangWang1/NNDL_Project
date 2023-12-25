from utils.trainer import train, evaluation
from utils.data_loader import dataloader
from utils.config import Config
import json

# nltk.download('punkt')


if __name__ == '__main__':
    config = Config()
    config.read_config('checkpoints/12-25_00-45CNN_Transformer/config.json')  # 读取参数，打印参数
    # 数据加载
    train_loader, test_loader = dataloader('data/deepfashion-mini', config.batch_size, workers=0)
    # 模型训练
    # train(train_loader, config)

    # 模型评估
    vocab_path = 'data/deepfashion-mini/vocab.json'
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    evaluation(test_loader, config, vocab)

