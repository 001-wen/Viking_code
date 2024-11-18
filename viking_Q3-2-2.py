import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# 定义残差块
class ResidualBlock(nn.Module):     #继承Module类
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 两个卷积层，卷积核大小为3，步幅为1，padding为1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)  #卷积
        self.bn1 = nn.BatchNorm2d(out_channels)     #输出特征归一化
        self.relu = nn.ReLU(inplace=True)   #使用ReLU激活函数
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)  #二次卷积
        self.bn2 = nn.BatchNorm2d(out_channels)     ##输出特征归一化
        self.shortcut = nn.Sequential()     # 跳跃连接：如果输入和输出通道数不一致，则用1x1卷积调整
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):   #前馈过程
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# 定义ResNet网络结构
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        # 初始卷积层，输入通道为3，输出通道为64，卷积核大小为7，步幅为2，填充为3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     #窗口大小为3，步幅为2
        # 残差块层，定义了四个残差模块
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        # 全局池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类层
        self.fc = nn.Linear(512, num_classes)
    #构建一个由多个残差块组成的层
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):       #前馈
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    train_losses = []  # 用来保存每个epoch的训练损失
    test_accuracies = []  # 用来保存每个epoch的测试准确率

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练一个epoch
        for i, (inputs, labels) in enumerate(train_loader):     # 将数据转移到设备上（GPU/CPU）
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 前馈
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # 反馈迭代
            optimizer.step()
            running_loss += loss.item()  # 累积损失
            _, predicted = torch.max(outputs.data, 1)  # 计算准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算训练集损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # 在测试集上评估模型
        test_acc = evaluate_model(model, test_loader)
        test_accuracies.append(test_acc)
    return train_losses, test_accuracies

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for inputs, labels in test_loader:
            # 将数据转移到设备上（GPU/CPU）
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# 绘制训练损失和测试准确率曲线
def plot_metrics(train_losses, test_accuracies):
    epochs = np.arange(1, len(train_losses) + 1)

    # 绘制训练损失
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # 绘制测试准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label="Test Accuracy", color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    # 数据预处理和增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 图像大小调整
        transforms.RandomHorizontalFlip(p=0.5),  # 数据增强：随机水平翻转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    train_dir = 'D:/data-picture/train'  # 训练数据路径
    test_dir = 'D:/data-picture/test'  # 测试数据路径
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)  # 加载数据集
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=2)

    # 判断是否可以使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    model = ResNet(num_classes=len(train_dataset.classes)).to(device)   # 加载模型并转移到指定设备（GPU或CPU）
    criterion = nn.CrossEntropyLoss()   ## 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)    # 定义优化器
    # 训练并记录损失和准确率
    num_epochs = 10
    train_losses, test_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs)

    plot_metrics(train_losses, test_accuracies)
