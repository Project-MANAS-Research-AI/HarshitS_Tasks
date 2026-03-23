import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# loading CIFAR-10 dataset
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# simple MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)  # flatten image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()



# optimizer = optim.SGD(model.parameters(), lr=0.01)

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

optimizer = optim.Adam(model.parameters(), lr=0.001)


criterion = nn.CrossEntropyLoss()

# training loop
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in trainloader:
        optimizer.zero_grad()   # clear gradients

        outputs = model(images) # forward pass
        loss = criterion(outputs, labels)

        loss.backward()         # backprop
        optimizer.step()        # update weights

        running_loss += loss.item()

    print("Epoch:", epoch+1, "Loss:", running_loss)