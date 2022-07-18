import math
nan = float("nan")
print(math.isnan(nan))

import torch
from torch import nn
t = torch.tensor([[1], [71955]], dtype = torch.float)
batch_size = 3
# t = torch.randn(batch_size, 1)
linear = nn.Linear(1,2)
print(linear.weight)
print(linear.bias)
logits = linear(t)
print(logits)

i = 0
for temp in logits:
    print(temp)
    if math.isnan(temp[0]) or math.isnan(temp[1]):
        print(f'{i} th occur nan')
        print(t[i])
    i = i + 1