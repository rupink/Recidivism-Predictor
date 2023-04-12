# -*- coding: utf-8 -*-
import torch.nn as nn
class MainNetwork(nn.Module):
  def __init__(self):
    super(MainNetwork, self).__init__()
    self.network = nn.Sequential(nn.Linear(444, 1), nn.Sigmoid(), nn.Linear(1, 1))
  def forward(self, x):
    result = self.network(x)
    return result

class Adversarial(nn.Module):
  def __init__(self):
    super(Adversarial, self).__init__()
    self.network = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
  def forward(self, x):
    result = self.network(x)
    return result


