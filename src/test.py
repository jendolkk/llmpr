import numpy as np
import torch
from torch.nn import LogSoftmax

B = 32
logits = torch.ones((B, B))
print(logits)
u = torch.arange(B)
print(u)