
import torch
import torch.nn as nn
import torch.optim as optim  #优化算法
import torchvision           #计算视觉工具
import torchvision.transforms as transforms #数据预处理
from torch.utils.data import DataLoader #数据加载器
import matplotlib.pyplot as plt #绘图库

# 定义神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 定义池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128) #7x7是经过两次池化后的特征图大小；隐藏层：3136->128
        self.fc2 = nn.Linear(128, 10) # 输出层：128->10(数字类别)
        self.dropout = nn.Dropout(0.5)# Dropout防止过拟合
        
    def forward(self, x):
        # 前向传播过程
        x = self.pool(torch.relu(self.conv1(x)))# 卷积1 + 激活 + 池化
        x = self.pool(torch.relu(self.conv2(x)))# 卷积2 + 激活 + 池化
        x = x.view(-1, 64 * 7 * 7)# 展平为一维向量
        x = torch.relu(self.fc1(x))# 全连接1 + 激活
        x = self.dropout(x)
        x = self.fc2(x)# 全连接2(输出)
        return x

def train_model():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),# 转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))# 标准化
    ])
    
    # 加载MNIST数据集
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    epochs = 5
    train_losses = []
    train_accuracies = []
    
    print("开始训练手写数字识别模型...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 遍历训练数据
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()     # 清零梯度
            output = model(data)      # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()           # 反向传播
            optimizer.step()          # 更新参数
            
            # 统计准确率
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 定期打印训练进度
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # 计算epoch统计信息
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    # 测试模型
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():# 禁用梯度计算
        for data, target in test_loader:
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    # 打印测试结果
    test_accuracy = 100. * test_correct / test_total
    print(f'测试集准确率: {test_accuracy:.2f}%')
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("模型已保存为 mnist_model.pth")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_curve.png')
    print("训练曲线已保存为 training_curve.png")
    
    return model

if __name__ == "__main__":
    model = train_model()
