import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= DATA =================
# Keeping preprocessing simple for now
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

# ================= MODEL =================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Trying a simple 3-layer MLP first
        self.fc1 = nn.Linear(32*32*3, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.5)  # might tune later

        self._init_weights()

    def _init_weights(self):
        # Using He init since I'm using ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc3(x)

        return x

# ================= CUSTOM OPTIMIZERS =================

class SGD:
    def __init__(self, params, lr=0.01, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data

            # Adding L2 regularization manually
            if self.weight_decay != 0:
                grad += self.weight_decay * p.data

            p.data -= self.lr * grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # velocity for each parameter
        self.velocities = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad.data

            if self.weight_decay != 0:
                grad += self.weight_decay * p.data

            # momentum update
            self.velocities[i] = self.momentum * self.velocities[i] + grad

            p.data -= self.lr * self.velocities[i]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad.data

            if self.weight_decay != 0:
                grad += self.weight_decay * p.data

            # Update biased estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# ================= LR SCHEDULER =================
def cosine_lr(initial_lr, epoch, max_epochs):
    # Trying cosine decay (seems smoother than step decay)
    return 0.5 * initial_lr * (1 + math.cos(math.pi * epoch / max_epochs))

# ================= TRAINING =================
def train(model, optimizer, epochs=20, initial_lr=0.001):
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Adjust LR each epoch
        lr = cosine_lr(initial_lr, epoch, epochs)
        optimizer.lr = lr

        print(f"\nEpoch {epoch+1} | LR = {lr:.6f}")

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # Small observation note
        if avg_val_loss > avg_train_loss:
            print("Observation: Possible overfitting starting.")

    return train_losses, val_losses

# ================= RUN =================
model = MLP().to(device)


optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# optimizer = SGD(model.parameters(), lr=0.01)
# optimizer = SGDMomentum(model.parameters(), lr=0.01)

train_losses, val_losses = train(model, optimizer, epochs=20)

# ================= PLOT =================
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

