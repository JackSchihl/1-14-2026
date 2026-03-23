import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

torch.manual_seed(42)

data = pd.read_csv('data.csv')
data['Diagnosis'] = data['Diagnosis'].map({'Benign': 0, 'Malignant': 1})

features = torch.tensor(data.drop('Diagnosis', axis=1).to_numpy(), dtype=torch.float32)
target = torch.tensor(data['Diagnosis'].to_numpy(), dtype=torch.float32).reshape(-1, 1)

# Compute feature-wise mean/std (use PyTorch dim argument)
fm = features.mean(dim=0, keepdim=True)
fs = features.std(dim=0, keepdim=True)
# avoid division by zero for constant features
fs[fs == 0] = 1.0

# Normalize features only. Do NOT normalize the target for BCEWithLogitsLoss (expects 0/1 labels)
X = (features - fm) / fs
Y = target

# Linear input size must match number of features
in_features = X.shape[1]
model = nn.Linear(in_features, 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 250

for epoch in range(epochs):
    optimizer.zero_grad()
    Yhat = model(X)
    loss = criterion(Yhat, Y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch}: {loss.item():.6f}")

torch.save({
    'fm': fm,
    'fs': fs,
    'parameters': model.state_dict(),
}, 'model.pth')
