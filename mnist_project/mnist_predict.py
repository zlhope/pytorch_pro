
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
from mnist_train import SimpleCNN
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
def predict_digit():
    # 加载训练好的模型
    model = SimpleCNN()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    # 数据预处理（与训练时一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("手写数字识别模型已加载，准备进行预测...")
    
    # 示例：创建一个简单的手写数字图像进行测试
    def create_test_image(digit):
        img = Image.new('L', (28, 28), color=0)  # 黑色背景
        draw = ImageDraw.Draw(img)
        
        # 根据数字绘制简单的图形
        if digit == 0:
            draw.ellipse([4, 4, 24, 24], outline=255, width=2)
        elif digit == 1:
            draw.line([14, 4, 14, 24], fill=255, width=3)
        elif digit == 2:
            # 手动绘制类似2的形状
            draw.line([6, 6, 22, 6], fill=255, width=2)     # 上横线
            draw.line([22, 6, 22, 14], fill=255, width=2)   # 右竖线
            draw.line([6, 14, 22, 14], fill=255, width=2)   # 中横线
            draw.line([6, 14, 6, 22], fill=255, width=2)    # 左竖线
            draw.line([6, 22, 22, 22], fill=255, width=2)   # 下横线
        elif digit == 3:
            # 绘制类似3的形状
            draw.line([6, 6, 22, 6], fill=255, width=2)     # 上横线
            draw.line([22, 6, 22, 14], fill=255, width=2)   # 右上竖线
            draw.line([6, 14, 22, 14], fill=255, width=2)   # 中横线
            draw.line([22, 14, 22, 22], fill=255, width=2)  # 右下竖线
            draw.line([6, 22, 22, 22], fill=255, width=2)   # 下横线
        elif digit == 4:
            # 绘制类似4的形状
            draw.line([12, 6, 12, 14], fill=255, width=2)   # 左上到中下的竖线
            draw.line([12, 14, 20, 14], fill=255, width=2)  # 横线
            draw.line([20, 6, 20, 22], fill=255, width=2)   # 右边的长竖线
        elif digit == 5:
            # 绘制类似5的形状
            draw.line([6, 6, 22, 6], fill=255, width=2)     # 上横线
            draw.line([6, 6, 6, 14], fill=255, width=2)     # 左上竖线
            draw.line([6, 14, 22, 14], fill=255, width=2)   # 中横线
            draw.line([22, 14, 22, 22], fill=255, width=2)  # 右下竖线
            draw.line([6, 22, 22, 22], fill=255, width=2)   # 下横线
        elif digit == 6:
            # 绘制类似6的形状
            # 画一个椭圆作为基础
            draw.ellipse([10, 10, 20, 20], outline=255, width=2)
            # 在椭圆的左侧加上一个垂直线，形成6的特征
            draw.line([12,12, 12, 0], fill=255, width=2)
        elif digit == 7:
            # 绘制类似7的形状
            draw.line([6, 6, 22, 6], fill=255, width=2)     # 上横线
            draw.line([22, 6, 14, 22], fill=255, width=2)   # 斜线
        elif digit == 8:
            # 绘制类似8的形状
            draw.ellipse([6, 6, 22, 14], outline=255, width=2)  # 上圆
            draw.ellipse([6, 14, 22, 22], outline=255, width=2) # 下圆
        elif digit == 9:
            # 画一个不完整的圆（从右上到左下）
            # 画一个椭圆作为基础
            draw.ellipse([6, 6, 15, 15], outline=255, width=2)
            # 在椭圆的右侧加上一个垂直线，形成9的特征
            draw.line([15, 10, 15, 60], fill=255, width=2)
        
        return img
    
    # 测试所有数字
    print("\n测试模型识别能力:")
    print("-" * 40)
    
    for digit in range(10):
        test_img = create_test_image(digit)
        input_tensor = transform(test_img).unsqueeze(0)  # 添加batch维度
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted].item()
            
            print(f"真实数字: {digit}, 预测数字: {predicted}, 置信度: {confidence:.4f}")
        
        # 显示图像
        if digit < 10:  # 显示10个数字的图像
            plt.figure(figsize=(4, 4))
            plt.imshow(test_img, cmap='gray')
            plt.title(f'真实: {digit}, 预测: {predicted}')
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    predict_digit()
