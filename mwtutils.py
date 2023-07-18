import torch
import torch.nn as nn

import numpy as np
from scipy.io import loadmat, savemat
import math
import os
import h5py
import torch.nn.functional as F
from functools import partial
# from models import MWT1d
from utils import train, test, LpLoss, get_filter, UnitGaussianNormalizer
from utilities3 import *
from timeit import default_timer
from typing import List, Tuple
from torch import Tensor
import scipy.io as sio

    
    
    
    
    
    
    
    
def get_initializer(name):
    '''Different Initializations'''

    if name == 'xavier_normal':
        init_ = partial(nn.init.xavier_normal_)
    elif name == 'kaiming_uniform':
        init_ = partial(nn.init.kaiming_uniform_)
    elif name == 'kaiming_normal':
        init_ = partial(nn.init.kaiming_normal_)
    return init_

