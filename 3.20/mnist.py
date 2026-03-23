import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from grid import save_image_grid

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform = transforms.ToTensor()
)
# image,label = dataset[0]
# image.save('image.png')

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle = True
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10


