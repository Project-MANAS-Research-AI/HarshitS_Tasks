import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
X = torch.tensor([[0.,0.],
                  [0.,1.],
                  [1.,0.],
                  [1.,1.]])

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2,4),
            nn.Tanh(),
            nn.Linear(4,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.model(x)

model = XORNet()

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# training
for epoch in range(5000):

    pred = model(X)

    loss = loss_fn(pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(epoch, loss.item())

# test
print("Predictions:")
print(model(X).round())