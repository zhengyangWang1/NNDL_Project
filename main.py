from utils.trainer import cts_train
from utils.trainer import cnn_gru_train
from utils.data_loader import dataloader
from utils.config import Config

# nltk.download('punkt')


if __name__ == '__main__':
    config = Config()
    config.read_config('config.json')  # 读取参数，打印参数
    # 数据加载
    train_loader, test_loader = dataloader('data/deepfashion-mini', config.batch_size, workers=0)

    # 模型训练
    cts_train(train_loader, config)
