from turtle import position
import torch
hidden_size = 4
position_size1 = 2
position_size2 = 4
embedding = torch.nn.Embedding(position_size1, hidden_size)# <class 'torch.nn.modules.sparse.Embedding'>
print(embedding.weight)# <class 'torch.nn.parameter.Parameter'>

tokens = torch.tensor([i for i in range(position_size1)])
print(embedding(tokens))

embedding2 = torch.nn.Embedding(position_size2, hidden_size)# <class 'torch.nn.modules.sparse.Embedding'>
print(embedding2.weight)# <class 'torch.nn.parameter.Parameter'>

with torch.no_grad():
    # for i in range(embedding2.weight.size()[0]):
    #     embedding2.weight[i] = torch.zeros(embedding2.weight.size()[1])
    # print(embedding2.weight)
    for i in range(embedding.weight.size()[0]):
        embedding2.weight[i] = embedding.weight[i]
    for i in range(embedding.weight.size()[0], embedding2.weight.size()[0]):
        embedding2.weight[i] = torch.zeros(embedding2.weight.size()[1])

print(embedding2.weight)