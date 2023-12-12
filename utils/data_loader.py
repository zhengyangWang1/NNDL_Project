# 读取数据，预处理数据
import os


def data_process(data_file='data/deepfashion-mini'):
    # 输入数据：12694张图片，训练集和测试集的json文件(10155, 2538)，json中包含{图片名：描述}

    image_data = os.path.join(data_file, 'images')  # 图片路径
    # print(image_data)
    train_json = os.path.join(data_file, 'train_captions.json')
    test_json = os.path.join(data_file, 'test_captions.json')


if __name__ == '__main__':
    data_process()
