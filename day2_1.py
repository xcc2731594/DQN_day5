import torch
import torch.nn as nn
from torch.autograd import Variable


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)

x = Variable(torch.Tensor([[0.1,0.8],[0.8,0.2]]))
y = Variable(torch.Tensor([[1],[0]]))

net = MyNet()
mls = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(),lr = 0.01)
for i in range(1000):
    out = net(x)
    loss = mls(out,y)
    opt.zero_grad()
    loss.backward()
    opt.step()

print(net(x))