import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
features = torch.tensor(data.drop('MPG', axis=1).to_numpy()).float()
target = torch.tensor(data['MPG'].to_numpy()).float()

fm = features.mean(axis = 0, keepdim=True) #keeps dimensions of the original data, helps professor sleep at night
fs = features.std(axis = 0, keepdim=True)
tm = target.mean(axis = 0, keepdim=True)
ts = target.std(axis = 0, keepdim=True)

X = (features - fm)/fs
Y = (target - tm)/ts

model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 1000
for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)

torch.save({
    'fm': fm,
    'fs': fs,
    'tm': tm,
    'ts': ts,
    'paramters': model.state_dict()
}, 'model.pth')

