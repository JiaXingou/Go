import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from predict import model_model
# 自定义Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        eps = 1e-7
        p_t = (target * input) + ((1 - target) * (1 - input))
        focal_loss = - (self.alpha * (1 - p_t) ** self.gamma) * torch.log(p_t + eps)
        return torch.mean(focal_loss)

# 模型定义（这是一个示例模型，你需要替换为你自己的模型）

# 创建模型和损失函数
model = model_model()
criterion = FocalLoss(gamma=2, alpha=0.25)

# 优化器定义
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据加载器示例（替换为你的数据加载器）
train_loader = DataLoader(your_training_data, batch_size=1, shuffle=True)

# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}')

# 训练完成后，你的模型将在PyTorch中使用Focal Loss和Adam优化器进行20个epoch的训练。
