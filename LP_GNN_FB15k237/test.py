import torch
import torch.nn as nn
import torch.nn.functional as F

A = torch.tensor([0, 1, 2, 3])
B = torch.tensor([4, 5, 6, 7])
t1, t2 = A.chunk(2)
t3, t4 = B.chunk(2)

reshape1 = torch.cat([t2, t1], dim=0)
reshape2 = torch.cat([t4, t3], dim=0)

  

