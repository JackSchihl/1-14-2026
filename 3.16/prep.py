import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from export import export_model
from torch.utils.data import Dataset, DataLoader



data = pd.read_csv('data.csv')
X = torch.tensor(data.drop('y', axis=1).values, dtype=torch.float32)
Y = torch.tensor(data['y'].values, dtype=torch.float32).view(-1, 1)

class my_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
dataset = my_dataset(X, Y)

loader = DataLoader(
        dataset,
        batch_size = 2
    )

for x,y in loader:
    print(x)

model = nn.Linear(2,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100

for range in range(epochs):
    for X,Y in loader:
        optimizer.zero_grad()
        Yhat = model(X)
        loss = criterion(Yhat, Y)
        loss.backward()
        optimizer.step()
    print(loss)