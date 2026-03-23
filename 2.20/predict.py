import torch
import torch.nn as nn

model_data = torch.load('model.pth')
fm = model_data['fm']
fs = model_data['fs']
tm = model_data['tm']
ts = model_data['ts']
parameters = model_data['paramters']

features = torch.tensor([])
print(parameters)