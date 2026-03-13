
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("data/gene_expression.csv")

X = df.drop("Cancer Present", axis=1).values
y = df["Cancer Present"].values

y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)   

    def forward(self, x):
        return self.linear(x)
input_dim = X_train.shape[1]
model = SVM(input_dim)
criterion = nn.HingeEmbeddingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,
1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted = torch.sign(test_outputs).view(-1)
    accuracy = (predicted == y_test_tensor.view(-1)).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")




