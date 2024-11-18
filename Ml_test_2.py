import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from PIL import Image


# 1. 定义残差模块（Residual Block）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        out += self.skip_connection(identity)
        return torch.relu(out)


# 2. 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.layer1 = self.make_layer(64, 128)
        self.layer2 = self.make_layer(128, 256)
        self.layer3 = self.make_layer(256, 512)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 3. 数据预处理与加载
# 数据增强与标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 以ImageNet标准均值和标准差
])

# 加载数据集并划分训练集和测试集
data_dir = 'path_to_dataset'  # 数据集路径，包含 train 和 test 子目录

# 使用ImageFolder加载数据集
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 5. 初始化模型、损失函数与优化器
model = ResNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 6. 训练模型
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # 评估模型的准确率
        test_acc = evaluate(model, test_loader)
        test_accuracies.append(test_acc)
        print(f"Test Accuracy: {test_acc:.4f}")

    return train_losses, test_accuracies


# 7. 测试模型并计算准确率
def evaluate(model, test_loader):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# 8. 可视化训练过程
def plot_metrics(train_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # 绘制损失函数
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')

    plt.show()


# 9. 训练并可视化结果
train_losses, test_accuracies = train(model, train_loader, criterion, optimizer, num_epochs=10)
plot_metrics(train_losses, test_accuracies)
