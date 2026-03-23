import torch
from matrepr import mprint
import numpy as np

A = torch.tensor([
    [2,7],
    [3,4]
])

B = torch.tensor([
    [1,2],
    [5,3]
])

mprint(A*B)