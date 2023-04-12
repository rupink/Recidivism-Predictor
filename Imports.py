# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math
import torch 
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation

LINK = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
CONTENTS = readlink = pd.read_csv(LINK)