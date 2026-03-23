#Finding the Line of Best Fit Using Gradient Descent in PyTorch
import torch

#Enter information about the data (x and y values of the points)
x = torch.tensor([
    [1.0],
    [5.0],
    [9.0]
    ])

y = torch.tensor([
    [5.0],
    [8.0],
    [2.0]
    ])

#Remember to set requires_grad=True for weights and biases
#as they have to be updated
w = torch.tensor([
    [0.0]
], requires_grad=True)

b = torch.tensor([
    [0.0]
    ], requires_grad=True)

#Set number of epochs and learning rate
epochs = 5000
lr = 0.01

#Perform gradient descent using the data provided and a for loop
for epoch in range(epochs):
    yhat = x @ w + b
    r = yhat - y
    SSE = r.T@r
    loss = SSE/3
    loss.backward()

#Update weights and biases
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad

    print(loss.item(), w, b)

#Zero out the gradients
    w.grad.zero_()
    b.grad.zero_()

