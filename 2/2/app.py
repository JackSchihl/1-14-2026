import torch

x = torch.tensor(3.0, requires_grad = True)
y = torch.tensor(3.0, requires_grad = True)
z = torch.tensor(2.0, requires_grad = True)

f = (-4*y**2 - 4*x**3*y - x) / (3*y + 1*x**2*y**2 + 5*y*x + 2)
f.backward()
print(x.grad, y.grad, z.grad)
