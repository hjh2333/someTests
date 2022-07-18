import math
from torch import nn
import torch
import random
# batch_size = 3
# class_num = 5
# output = torch.randn(batch_size, class_num, requires_grad=True)
# label = torch.empty(batch_size, dtype=torch.long).random_(class_num)
# # weights = torch.FloatTensor([2, 3, 11, 12, 13])
# weights = torch.randn(class_num)
batch_size = 1
class_num = 2
output = torch.FloatTensor([[0.0445, 0.2135]]).view(-1,class_num)
label = torch.tensor([1]).long()
weights = torch.FloatTensor([1, 2])
criterion = nn.CrossEntropyLoss(weight=weights)
loss = criterion(output, label)

print(f"网络输出为{batch_size}样本{class_num}类别:")
print(output)
print("要计算loss的类别:")
print(label)
print("计算loss的结果:")
print(loss)

first = [0] * batch_size
for i in range(batch_size):
    first[i] = -output[i][label[i]]
second = [0] * batch_size
for i in range(batch_size):
    for j in range(class_num):
        second[i] += math.exp(output[i][j])
res = 0
for i in range(batch_size):
    res += weights[label[i]]*(first[i] + math.log(second[i]))
weis = 0
for i in range(batch_size):
    weis += weights[label[i]]

print("自己的计算结果：")
print(res / weis) # nn.CrossEntropyLoss权重是对整个样本算的，而不是/weights.sum()

'''
2022-07-04 19:45:52 PM : DEBUG : weight
2022-07-04 19:45:52 PM : DEBUG : tensor([1., 2.])
2022-07-04 19:45:52 PM : DEBUG : input
2022-07-04 19:45:52 PM : DEBUG : tensor([[-0.0594,  0.3967],
        [-0.4912,  0.5791]], device='cuda:0', dtype=torch.float16,
       grad_fn=<AddmmBackward0>)
2022-07-04 19:45:52 PM : DEBUG : target
2022-07-04 19:45:52 PM : DEBUG : tensor([0, 1], device='cuda:0')
2022-07-04 19:45:52 PM : DEBUG : loss
2022-07-04 19:45:52 PM : DEBUG : tensor([0.9468, 0.5898], device='cuda:0', grad_fn=<SqueezeBackward0>)
'''