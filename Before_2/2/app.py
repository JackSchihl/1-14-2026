import torch
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

X = torch.tensor(data.drop('Y', axis = 1).to_numpy()).float()
Y = torch.tensor(data['Y'].to_numpy()).float().reshape(-1, 1)

w = torch.tensor([
    [1.40],
    [2.90],
    [2.90],
    [-2.50],
    [1.30]
]).float()

b = torch.tensor([
    [-2.30]
]).float()

Yhat = X@w + b
r = Yhat - Y
SSE = r.T@r
loss = SSE/3
print(loss)