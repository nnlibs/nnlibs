import torch 
import torch.nn as nn

relu = nn.ReLU()

input = torch.tensor([[1,2,-3],[4,-5,6]],dtype=torch.float32,requires_grad=True)

output = relu(input)

print(output)

output_grad = torch.tensor([[1,1,1],[1,1,1]],dtype=torch.float32)
output.backward(output_grad)
print(input.grad)

