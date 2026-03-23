import torch

X = torch.tensor(4)
w = torch.tensor(3)
b = torch.tensor(10)
Y = torch.tensor(13)

Yhat = X*w+b
loss = (Yhat - Y)**2
print(f"Yhat = {Yhat}")
print(f"loss = {loss}")

X = torch.tensor([
    [4],
    [6]
])
w = torch.tensor(3)
b = torch.tensor(10)
Y = torch.tensor([
    [5],
    [6]
])
Yhat = X*w+b
r = Yhat - Y
SSE = r.T@r
loss = SSE/2
print(f"loss = {loss}")