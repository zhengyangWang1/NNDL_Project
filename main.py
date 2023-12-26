from utils.trainer import train, evaluate, evaluation
from utils.data_loader import get_dataloader, data_preprocess
from utils.config import Config
import json

# nltk.download('punkt')


if __name__ == '__main__':
    config = Config()
    config.read_config('config.json')  # 读取参数，打印参数
    # 数据加载
    # data_preprocess(config.dataset_path)
    train_loader, test_loader,eval_loader = get_dataloader(config.dataset_path,
                                               config.batch_size,
                                               config.eval_batch_size,
                                               workers=0)
    # 模型训练
    train(train_loader, test_loader, config)
    # evaluate(test_loader, config, model_checkpoint_path='checkpoints/12-25_23-08CNN_Transformer/model_checkpoint2.pth')

    # 模型评估

    # evaluation(test_loader, config)
