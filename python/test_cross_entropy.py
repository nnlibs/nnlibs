import torch.nn as nn
import torch

ce = nn.CrossEntropyLoss()

output = torch.tensor([[0.1, 0.82, 0.08]],requires_grad=True)

target = torch.tensor([2])

loss = ce(output, target)

loss.backward()

print(loss)
print(output.grad)

