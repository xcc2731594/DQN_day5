import torch
import torch.nn as nn
from itertools import repeat
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128*32*32,64),
            nn.ReLU(),
            nn.Linear(64,48),
            nn.ReLU(),
            nn.Linear(48,10)
        )
    def forward(self,inputs):
        out = self.con1(inputs)
        out = self.con2(out)
        out = out.view(1,-1)


x = torch.Tensor([1,2,3])
print(torch.argmax(x).item())