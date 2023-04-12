# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:39:34 2023

@author: rupin
"""
import torch.nn as nn
import torch

class LogisticRegression(nn.Module):
  def __init__(self, n_input, n_output):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(n_input, n_output)
  def forward(self,x):
    result = torch.sigmoid(self.linear(x))
    return result
