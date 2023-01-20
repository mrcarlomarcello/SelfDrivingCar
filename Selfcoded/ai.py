#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:47:59 2023

@author: marcellomenjivarmontesdeoca
"""

#importing the libraries that we need

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creamos la architectura de la Red Neuronal(Neural Network)

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
#implementing experience replay for memory purposes

class ReplayMemory(object):
    
    def __init__(self, _capacity):
        self.capacity = _capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)