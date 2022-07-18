from attr import has
import torch.nn as nn
class test(nn.Module):
    h = 2
    def __init__(self, input):
        super(test,self).__init__()
        self.input = input

    def forward(self,x):
        return self.input * x

    def hello(self,x):
        return x

T = test(8)
print(T(6))
print(T.input)
print(T.h)
print(hasattr(test, "h"))

T2 = test(8)
print(T2(6))
print(T2.input)
print(T2.h)
print(hasattr(T2, "input"))