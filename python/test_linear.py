import torch
import torch.nn as nn
import torch.optim as optim

# 设置输入输出特征维度
out_features = 5
in_features = 10
learning_rate = 0.001

# 定义线性层并设置权重和偏置
linear = nn.Linear(in_features, out_features, bias=True)

# 手动设置权重和偏置值
with torch.no_grad():
    linear.weight = nn.Parameter(torch.tensor([
        [1,  1, 0, 1, 1,  1, 0, 0,  1,  0],
        [-1, 0, 1, 1, -1, 0, 1, 1, 1,  0],
        [1,  -1, 0, 1, -1,  1, 1, 1, 1,  0],
        [2,  0, 1, 2, -1, 1, 1,  1, 1,  0],
        [0,  0, -1, 2, 1, 2, 1,  1,  1,  0]
    ], dtype=torch.float32))
    linear.bias = nn.Parameter(torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32))

# 定义输入张量
input_tensor = torch.tensor([[i + 1 for i in range(in_features)]], dtype=torch.float32)

optimizer = optim.SGD(linear.parameters(), lr=learning_rate)
# 前向传播
output = linear(input_tensor)
print("Output:", output)
grad_output = torch.tensor([[1, 0, 0, 1, 1]], dtype=torch.float32)
output.backward(grad_output)
print("Gradients of input_data:\n", input_tensor.grad)
print("Gradients of weights:\n", linear.weight.grad)
print("Gradients of bias:\n", linear.bias.grad)

optimizer.step()
print("Gradients of weights:\n", linear.weight)
print("Gradients of bias:\n", linear.bias)

