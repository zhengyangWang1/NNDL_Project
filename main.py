import torch.optim as optim

from utils.data_loader import dataloader
from model import Model
from model import ResNetEncoder, GRUDecoder, PackedCrossEntropyLoss

if __name__ == '__main__':
    encoder = ResNetEncoder()
    decoder = GRUDecoder(2048, 512, 300, 512, num_layers=1)
    model = Model(encoder, decoder)
    train_data, test_data = dataloader('data/deepfashion-mini', 8, workers=0)

    # 定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = PackedCrossEntropyLoss()

    # 迭代训练
    num_epochs=10
    for epoch in range(num_epochs):  # num_epochs 为训练轮数
        num_sample = 0
        running_loss = 0.0
        for i, (imgs, caps, caplens) in enumerate(train_data):
            # 获取输入数据
            optimizer.zero_grad()
            grid = encoder(imgs)
            predictions, sorted_captions, lengths, sorted_cap_indices = decoder(grid, caps, caplens)
            loss = loss_fn(predictions, sorted_captions[:, 1:], lengths)
            num_sample += imgs.shape[0]
            running_loss += loss * imgs.shape[0]
            loss.backward()
            optimizer.step()
        average_loss = running_loss / num_sample
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
