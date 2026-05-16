"""
手机价格预测案例

执行步骤：
1. 获取数据集，对数据集进行处理   数据 -> 张量(Tensor) -> 张量数据集(TensorDataset) -> 数据加载器(DataLoader)
2. 构建神经网络，写__init__ 和 forward 函数
3. 模型训练
4. 模型测试

根据数据集，提取特征并建立一个 20 -> 256 -> 512 -> 4的一个神经网络
通过训练 -> 保存模型参数 -> 验证 得到一个模型和准确率
对模型进行调优
1. 使用Adam优化策略
2. 调整学习率
3. 使用BN优化/对数据进行标准化处理
4. 增加网络深度
5. 调整训练轮数
.....

"""
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler


# todo: 1. 获取数据集，对数据集进行处理
def get_dataset():
    # 获取数据集
    data = pandas.read_csv("./data/手机价格预测.csv")
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x_data = x.astype(np.float32)
    y_data = y.astype(np.int64)
    # 划分训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # 对数据进行标准化处理，调优策略
    transfer = StandardScaler()
    train_x = transfer.fit_transform(train_x)
    test_x = transfer.transform(test_x)
    # 数据转为张量 -> 张量数据集
    data_train = TensorDataset(torch.tensor(train_x), torch.tensor(train_y.values))
    data_test = TensorDataset(torch.tensor(test_x), torch.tensor(test_y.values))

    return data_train, data_test, train_x.shape[1], len(np.unique(y))


# todo: 2. 构建神经网络，写__init__ 和 forward 函数
class PhonePriceNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PhonePriceNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512) # 调优
        self.output = nn.Linear(512, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.output(x)
        return x


# todo: 3. 模型训练
def train(data_train, input_size, output_size):
    # 1. 创建模型
    model = PhonePriceNetwork(input_size, output_size)
    # 2. 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 3. 定义优化器
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 调优
    # 4. 创建数据加载器
    data_loader = DataLoader(data_train, batch_size=16, shuffle=True)
    # 5. 定义所需参数
    # epochs = 50
    epochs = 70  # 调优
    for epoch in range(epochs):
        total_loss, batch_num = 0.0, 0
        start_time = time.time()
        for x, y in data_loader:
            # 1. 调整模型状态
            model.train()
            # 2. 前向传播
            y_predict = model(x)
            # 3. 计算损失
            loss = criterion(y_predict, y)
            # 4. 反向传播
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            # 5. 累计损失
            total_loss += loss.item()
            batch_num += 1
        print(f"训练轮数：{epoch + 1}, 平均损失：{total_loss / batch_num:.4f}, 训练时间：{time.time() - start_time:.2f}s")

    print("训练完成，保存模型中...")
    os.makedirs("./model", exist_ok=True)
    # 保存的模型参数文件类型：.pth, .pkl, .pickle
    torch.save(model.state_dict(), "./model/phone_price.pth")
    print("保存完成")


# todo: 4. 模型测试
def evaluate(data_test, input_size, output_size):
    # 1. 加载模型
    model = PhonePriceNetwork(input_size, output_size)
    model.load_state_dict(torch.load("./model/phone_price.pth"))
    # 2. 创建数据加载器
    data_loader = DataLoader(data_test, batch_size=16, shuffle=False)
    # 3. 进行测试
    correct = 0
    for x, y in data_loader:
        # 1. 设置模型状态
        model.eval()
        # 2. 预测目标值,
        output = model(x)
        # 3. 获取类别结果，由于神经网络没有使用softmax且输出的内容为概率，所以需要用argmax获取最大概率对应的下标, dim=1表示按照行计算
        y_pred = torch.argmax(output, dim=1)
        # 4. 获取预测结果正确的数量
        correct += (y == y_pred).sum()
    print(f"准确率：{correct.item() / len(data_test):.4f}")


if __name__ == '__main__':
    data_train, data_test, input_size, output_size = get_dataset()
    # print("数据集大小：", len(data_train), len(data_test))
    # print("数据集特征维度：", input_size)
    # print("数据集标签维度：", output_size)
    train(data_train, input_size, output_size)
    evaluate(data_test, input_size, output_size)
