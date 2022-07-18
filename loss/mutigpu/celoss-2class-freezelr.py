'''
CELoss
'''
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

input = torch.arange(64).view(2,-1).float()
ground_truth1 = torch.tensor([0, 1]).long()
ground_truth2 = torch.tensor([0, 1]).long()


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(32, 8, bias=False)
        self.layer2 = nn.Linear(8, 2, bias=False)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return F.sigmoid(out2), F.sigmoid(out2)


model = MyModel()
model.train()

optimizer = optim.SGD(model.parameters(), 0.01)
output1, output2 = model(input)
print('output')
print(output1, output2)

loss1 = F.cross_entropy(output1, ground_truth1, reduction='mean')
loss2 = F.cross_entropy(output2, ground_truth2, reduction='mean')
print('loss')
print(loss1, loss2)

optimizer.zero_grad()
loss1.backward(retain_graph=True)
loss2.backward(retain_graph=True)
print(model.layer1.weight.grad[0][0])

optimizer.zero_grad()
loss = (loss1 + loss2)
loss.backward()
print(model.layer1.weight.grad[0][0])
