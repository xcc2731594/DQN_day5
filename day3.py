import torch
import torch.nn as nn

x = torch.Tensor([[0.2,0.4],[0.2,0.3],[0.3,0.4]])
y = torch.Tensor([[0.6],[0.5],[0.7]])

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,1)
        )
        self.opt = torch.optim.Adam(self.parameters())
        self.mls = torch.nn.MSELoss()

    def forward(self, inputs):
        return self.fc(inputs)

    def train_model(self,x,y):
        out = self.forward(x)
        loss = self.mls(out,y)
        print("loss:",loss)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
    def test(self,x):
        return self.forward(x)

net = MyNet()
for i in range(5000):
    net.train_model(x,y)

x = torch.Tensor([[0.5,0.1]])
out = net.test(x)
print(out)


