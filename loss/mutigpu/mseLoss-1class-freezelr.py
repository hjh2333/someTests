'''
https://www.zhihu.com/question/398425328/answer/1454276131
多任务学习中loss多次backward和loss加和后backward没区别（注意如果有ADAMW在优化学习率的话那未必一样）
'''
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

input = torch.rand((1, 64)).float()
ground_truth1 = torch.tensor([[0]]).float()
ground_truth2 = torch.tensor([[0]]).float()


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(64, 8, bias=False)
        self.layer2 = nn.Linear(8, 1, bias=False)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return F.sigmoid(out2), F.sigmoid(out2)


model = MyModel()
model.train()

optimizer = optim.SGD(model.parameters(), 0.01)
output1, output2 = model(input)
print(output1, output2)

loss1 = F.mse_loss(output1, ground_truth1)
# loss1 = F.cross_entropy(output1, ground_truth1)
loss2 = F.mse_loss(output2, ground_truth2)
# loss2 = F.cross_entropy(output2, ground_truth2)

optimizer.zero_grad()
loss1.backward(retain_graph=True)
loss2.backward(retain_graph=True)
print(model.layer1.weight.grad[0, 0])

optimizer.zero_grad()
loss = (loss1 + loss2)
loss.backward()
print(model.layer1.weight.grad[0, 0])
