from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),       #大小调整
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),     #归一化
    transforms.RandomHorizontalFlip(p=0.5),     #数据增强
    transforms.ToTensor()   #转换为张量形式
])
#加载数据集
train_dir = 'D:/data-picture/train'     #数据集路径
test_dir = 'D:/data-picture/test'
train_dataset = datasets.ImageFolder(root=train_dir,transform=transform)    #加载数据集
test_dataset = datasets.ImageFolder(root=test_dir,transform=transform)
train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=4)      #创建loader
test_loader = DataLoader(test_dataset,batch_size=8,shuffle=True,num_workers=4)

import torch
if torch.cuda.is_available():
    torch.device("cuda")    #使用GPU加载

import torch.nn as nn
import torch.optim as optim

#定义残差块
class ResidualBlock(nn.Module):     #
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出的通道不一致，需要通过 1x1 卷积来匹配
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 跳跃连接
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差块（通过重复添加 ResidualBlock）
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # 全局池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类层
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
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


# 加载ResNet模型（通过调整输出类别数量）
model = ResNet(num_classes=len(train_dataset.classes)).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练和评估函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练一个 epoch
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 累积损失
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
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


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# 训练模型并记录损失和准确率
num_epochs = 10
train_losses, test_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs)

import matplotlib.pyplot as plt
import numpy as np
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


# 可视化训练过程
plot_metrics(train_losses, test_accuracies)




def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    train_losses = []  # 用来保存每个epoch的训练损失
    test_accuracies = []  # 用来保存每个epoch的测试准确率

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练一个epoch
        for i, (inputs, labels) in enumerate(train_loader):
            # 将数据转移到设备上（GPU/CPU）
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # 反向传播和优化
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

# 训练并记录损失和准确率
num_epochs = 10
train_losses, test_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs)

# 绘制训练损失和测试准确率曲线
def plot_metrics(train_losses, test_accuracies):
    epochs = np.arange(1, len(train_losses) + 1)

    # 绘制训练损失
    fig1=plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # 绘制测试准确率
    # plt.subplot(1, 2, 2)
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(epochs, test_accuracies, label="Test Accuracy", color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()