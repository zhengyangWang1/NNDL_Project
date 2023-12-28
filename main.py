import json
import torch

from utils.trainer import train, evaluate, evaluation
from utils.data_loader import get_dataloader, data_preprocess
from utils.config import Config
from model.model import CNNTransformerModel
from utils.metrics import evaluate_metrics,metrics_calc
# from nlgeval.nlgeval

# nltk.download('punkt')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # 生成句子测试
    # evaluation(test_loader, config)

    # 加载config和模型
    # config.read_config('config.json')

    # checkpoint = torch.load('checkpoints/12-25_23-08CNN_Transformer/model_checkpoint2.pth')
    # model = CNNTransformerModel(vocab_size=config.vocab_size,
    #                             embed_size=config.CNN_Transformer.embed_size,
    #                             num_head=config.CNN_Transformer.num_head,
    #                             num_encoder_layer=config.CNN_Transformer.num_decoder,
    #                             num_decoder_layer=config.CNN_Transformer.num_decoder,
    #                             dim_ff=config.CNN_Transformer.dim_ff, ).to(device)
    # model.load_state_dict(checkpoint['model'])
    # model.eval()
    # evaluate_metrics(eval_loader,model,config)

    # metrics_calc('data/textout')
