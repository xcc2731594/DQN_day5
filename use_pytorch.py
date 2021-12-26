import torch
import torch.nn as nn
import matplotlib.pyplot as plt
X = torch.tensor([1.,2,3,4,5]).reshape(-1,1)
Y = torch.tensor([1.,2.5,3,4,5]).reshape(-1,1)
model = nn.Linear(1,1)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(),lr=0.0001)
for epotch in range(5000):
    for x,y in zip(X,Y):
        y_pred = model(x)
        loss = loss_fn(y,y_pred)
        opt.zero_grad()
        loss.backward()
        opt.step()

print(model.weight)
plt.scatter(X.numpy(),Y.numpy())
plt.plot(X.numpy(),model(X).detach().numpy())
plt.show()