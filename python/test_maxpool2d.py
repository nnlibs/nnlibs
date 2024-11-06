import torch.nn as nn
import torch

mp = nn.MaxPool2d(2, stride=2)

input = torch.tensor([[[[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]]]],dtype=torch.float32,requires_grad=True)

output = mp(input)
print(input)
print(output)

a = torch.tensor([[[[1, 2],
                            [3, 4]]]],dtype=torch.float32,requires_grad=True)
output.backward(a)
print(input.grad)
