import torch
import torch.nn as nn












X =(features - fm)/fs

model = nn.Linear(1,1)
model.load_state_dict(parameters)

prediction = model(X)
print(prediction*ts + tm)