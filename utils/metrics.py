import os
import torch
import torch.nn as nn
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate import meteor, bleu
from nlgeval import NLGEval
from pprint import pprint

def filter_useless_words(sent, filterd_words):
    # 去除句子中不参与BLEU值计算的符号
    return [w for w in sent if w not in filterd_words]


# https://www.nltk.org/api/nltk.translate.bleu_score.html
def evaluate_metrics(eval_loader, model, config):
    """
    :param eval_loader: eval_loader只加载图像和 一堆句子的loader
    :param model:  模型
    :param config:
    :return:
    """
    # 加载词典
    vocab, i2t = config.read_vocab()
    # 加载运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 评估模式
    model.eval()
    # model = CNNTransformerModel(vocab_size=config.vocab_size,
    #                             embed_size=config.CNN_Transformer.embed_size,
    #                             num_head=config.CNN_Transformer.num_head,
    #                             num_encoder_layer=config.CNN_Transformer.num_decoder,
    #                             num_decoder_layer=config.CNN_Transformer.num_decoder,
    #                             dim_ff=config.CNN_Transformer.dim_ff, ).to(device)
    # model.load_state_dict(checkpoint['model'])
    # 存储候选文本
    hyps = []
    # 存储参考文本
    refs = []
    # 需要过滤的词 集合
    filterd_words = {vocab['<start>'], vocab['<end>'], vocab['<pad>']}
    # 每个图像对应文本
    # cpi = config.captions_per_image
    device = next(model.parameters()).device
    with torch.no_grad():
        cnt = 0  # test
        tqdm_param = {
            'total': len(eval_loader),
            'mininterval': 0.5,
            'dynamic_ncols': True,
        }
        with tqdm(enumerate(eval_loader), **tqdm_param, desc='bleu-4') as t:
            for i, (imgs, caps, caplens) in t:
                cnt += 1  # test
                # 通过束搜索，生成候选文本 return 512条list的list
                # texts = model.beam_search(imgs.to(device), config.beam_k, config.max_len + 2, vocab)
                texts = model.greedy_search(imgs.to(device), config.max_len + 2, vocab)
                # 候选文本 512条语句
                hyps.extend([filter_useless_words(text, filterd_words) for text in texts])
                # 参考文本 caps 是tensor 需要转换成list 512,7,25 转换成list
                for cap in caps.tolist():  # cap是七条语句 循环512次
                    refs.append([filter_useless_words(c, filterd_words) for c in cap])
                # if cnt >= 1:
                #     break  # test
    # 实际上，每个候选文本对应cpi条参考文本

    # -------------------- 评估指标计算 --------------------
    print(len(refs), len(hyps))
    print(len(refs[1]), len(hyps[1]))
    # 形状 refs list 512batch list 7句子 list token
    # hyps list 512batch list token
    # TODO 转换为字符串类型计算
    for sens in range(len(refs)):
        for sen in range(len(refs[sens])):
            for token in range(len(refs[sens][sen])):
                refs[sens][sen][token] = i2t[refs[sens][sen][token]]
    for sen in range(len(hyps)):
        for token in range(len(hyps[sen])):
            hyps[sen][token] = i2t[hyps[sen][token]]
    # 转换为nlgeval需要的格式
    # ref = [reflist1,reflist2,...,reflist7] # 在字符串列表 不是token列表 去掉空格的列表
    # hyp = []
    # hyp_list = hyp
    # ref_list = [ref1, ref2]
    # res = nlge.compute_metrics(ref_list, hyp_list)
    # 去除标点
    filter_words = {',', '.', }
    eval_refs = []
    for sen in range(len(refs[0])):
        eval_refs.append([])  # 创建caption_per_image个空列表供后续使用
    for sens in range(len(refs)):  # 遍历所有样本 每条样本有七条句子
        for sen in range(len(refs[0])):  # 遍历七条句子 将句子转换为字符串加入到评估列表中
            eval_refs[sen].append(' '.join([x for x in refs[sens][sen] if x not in filter_words]))  # 将字符串列表合并为在一起的字符串
    eval_hyps = []
    for sen in range(len(hyps)):
        eval_hyps.append(' '.join([x for x in hyps[sen] if x not in filter_words]))
    # TODO 检查

    # 保存为文件
    os.makedirs('data/textout', exist_ok=True)
    for sen in range(len(eval_refs)):
        with open(f"data/textout/refs{sen}.txt", 'w+') as f:
            for sentence in eval_refs[sen]:
                f.write(sentence + '\n')
    with open(f"data/textout/hyps.txt", 'w+') as f:
        for sentence in eval_hyps:
            f.write(sentence + '\n')

    # TODO
    # from nlgeval import NLGEval
    # nlgeval = NLGEval()  # loads the models
    # metrics_dict = nlgeval.compute_metrics(eval_refs, eval_hyps)
    # print(f'Metrics Score:\n')
    # print(metrics_dict)

    # print(hyps[1])
    # print(hyps[2])
    # print(hyps[3])
    # # bleu分数 0-1之间越大越好 1-gram到4-gram的BLEU几何平均值
    # bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25))
    # # meteor分数
    # # TODO 需要下载wordnet
    # # import nltk
    # # nltk.download('wordnet')
    # meteor = 0
    # for i in range(len(refs)):
    #     meteor += meteor_score(refs[i], hyps[i])
    # meteor = meteor / len(refs)
    # # 其他评估指标实现，rouge-l, CIDEr-D,SPICE 后两个比较重要
    # model.train()
    # print(f'Metric Score: bleu-4:{bleu4}')
    # print(f'Metric Score: meteor:{meteor}')
    # return metrics_dict


def metrics_calc(text_path):
    cnt = 0
    prefix = 'ref'
    for filename in os.listdir(text_path):
        if filename.startswith(prefix):
            cnt += 1
    # eval_hyps = []

    eval_refs = []
    with open(os.path.join(text_path, 'hyps.txt')) as f:
        hyps = f.readlines()
        eval_hyps = [x.strip() for x in hyps]
    for i in range(cnt):
        with open(os.path.join(text_path, f'refs{i}.txt')) as f:
            refs = f.readlines()
            eval_refs.append([x.strip() for x in refs])

    # num_eval = 2530
    # eval_hyps = eval_hyps[:num_eval]
    # eval_refs = [x[:num_eval] for x in eval_refs]

    nlgeval = NLGEval(no_glove=True,no_skipthoughts=True)  # loads the models
    metrics_dict = nlgeval.compute_metrics(eval_refs,
                                           eval_hyps,
                                           )
    # noglove
    print(f'Metrics Score:')
    pprint(metrics_dict)
    return metrics_dict
