import torch
Xm = torch.tensor([
    [5.0]
    ])
Xs = torch.tensor([
    [4.0]
    ])
Xraw = torch.tensor([
    [7.0]
    ])
X = (Xraw - Xm) / Xs

w = torch.tensor([
    [-.5]
])

b = torch.tensor([
    [0.0]
    ])

pred = X @ w + b
print(pred)
