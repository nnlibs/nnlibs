import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
        # [
        #     [[-1, 1, 0], [0, 1, 0], [0, 1, 1]],
        #     [[-1, -1, 0], [0, 0, -1], [0, -1, 0]],
        #     [[-1, 1, 0], [1, 0, 1], [1, -1, -1]]
        # ],
        # [
        #     [[1, 1, -1], [-1, 1, 0], [-1, 0, -1]],
        #     [[-1, 1, 0], [-1, 0, 0], [0, -1, 0]],
        #     [[1, -1, 0], [1, 0, 0], [-1, 0, -1]]
        # ]

# 设置输入数据
input_data = torch.tensor([
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 2, 0],
        [0, 2, 2, 2, 2, 1, 0],
        [0, 1, 0, 0, 2, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 2, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [0, 1, 2, 1, 2, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 2, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 2, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 2, 1, 0, 1, 0],
        [0, 0, 1, 2, 2, 2, 0],
        [0, 2, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
], dtype=torch.float32).unsqueeze(0) # Shape: [1, 3, 7, 7]

# 定义目标梯度（grad_output）
grad_output = torch.tensor([
    [
        [0, 1, 1],
        [2, 2, 2],
        [1, 0, 0]
    ],
    [
        [1, 0, 2],
        [0, 0, 0],
        [1, 2, 1]
    ]
], dtype=torch.float32).unsqueeze(0) # Shape: [1, 2, 3, 3]

# 定义卷积层
conv = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=2, padding=0, bias=True)

# 手动设置卷积层的权重和偏置
with torch.no_grad():
    conv.weight = nn.Parameter(torch.tensor([
        [
            [[-1, 1, 0], [0, 1, 0], [0, 1, 1]],
            [[-1, -1, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 0, -1], [0, 1, 0], [1, -1, -1]]
        ],
        [
            [[1, 1, -1], [-1, -1, 1], [0, -1, 1]],
            [[0, 1, 0], [-1, 0, -1], [-1, 1, 0]],
            [[-1, 0, 0], [-1, 0, 1], [-1, 0, 0]]
        ]
    ], dtype=torch.float32))
    conv.bias = nn.Parameter(torch.tensor([1, 0], dtype=torch.float32))

learning_rate = 0.001
optimizer = optim.SGD(conv.parameters(), lr=learning_rate)

# 执行前向传播
output = conv(input_data)
print("Output:\n", output)

# 执行反向传播
output.backward(grad_output)
print("Gradients of input_data:\n", input_data.grad)
print("Gradients of weights:\n", conv.weight.grad)
print("Gradients of bias:\n", conv.bias.grad)

optimizer.step()
print("Gradients of weights:\n", conv.weight)
print("Gradients of bias:\n", conv.bias)
