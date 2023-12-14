
## 说明

### data文件夹结构

```text
data/
    deepfashion-mini/
    deepfashion/
```


### 数据处理：

原始数据为图片和json文件，处理为
data = {'IMAGES': img_paths, 'CAPTIONS': enc_captions}, 前者为图片路径, 后者为图片描述(转换为向量)

数据集被定义为Pytorch中的数据类Dataset

获得数据：
调用utils/data_loader中的函数(暂命名为)'Dataloader', 返回值为train_loader, test_loader(暂无验证集)
