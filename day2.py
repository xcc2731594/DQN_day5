import torch
import torch.nn as nn
from torch.autograd import Variable

x = [[0.1,0.8 ,1],[0.8,0.2,1]]
y = [[1],[0]]
w = [[0.1,0.2,0.3]]
x = Variable(torch.Tensor(x),requires_grad=False)
y = Variable(torch.Tensor(y),requires_grad=False)
w = Variable(torch.Tensor(w),requires_grad=True)

for i in range(1000):
    out = torch.mm(x,w.t())
    delta = (out-y)
    loss = delta[0] ** 2 + delta[1]**2
    print(loss)
    w.grad = torch.Tensor([[0,0,0]])
    loss.backward()
    w.data -= w.grad * 0.01
    
print("w.grad:",w.grad)
print(w.data)
print(torch.mm(x,w.t()))
