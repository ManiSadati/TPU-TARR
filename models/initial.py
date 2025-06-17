import torch.nn as nn
import torch


import matplotlib.pyplot as plt
import numpy as np

import math


model_type = 'resnet' #vgg,alexnet,resnet

dataset = 'cifar10'#'cifar100'

num_class = 10#100

log_level = 1 #1,5,10


prob = 1e-5
fault_check = True

alpha = torch.nn.Parameter(torch.tensor(5.0)) # create a tensor out of alpha
beta = torch.nn.Parameter(torch.tensor(6.0))

def log_act(alpha,beta):
    if log_level >= 10:
        plot(alpha,beta)
    elif log_level == 5:
        print(alpha,beta)
        input()
        #print('sum_alpha',torch.sum(alpha<2.8))
        #print('sum_beta',torch.sum(beta<1.5))
        #input()


model_name = 'ckpt_'+model_type+dataset

learning_rate = 1e-2

resume = True

model_bound = 'Layer'#Neuron,Layer
max_list = []

list_num = 0

def hook(self, input, output):
  if model_bound =='Layer':
    out = torch.max(output)
  elif model_bound =='Neuron':
    out,_ = torch.max(output,dim = 0)
  if(hook.cnt<list_num):
    max_list.append(out)
  else:
    mod = hook.cnt%list_num
    max_list[mod] = torch.max(max_list[mod],out)
  hook.cnt += 1
hook.cnt = 0

def register(net):
    global list_num
    for param in net.modules():
        if (isinstance(param, nn.ReLU)):
            list_num += 1
            param.register_forward_hook(hook)
            print('in register: ',list_num)


def bound_relu(self, input, output):
  #print('here')
  mode = bound_relu.cnt%list_num
  #print(mode)
  output1 = ((output <= max_list[mode]) & (output > 0.)) * output
  output[:,:,:,:] = output1[:,:,:,:] 
  bound_relu.cnt += 1
bound_relu.cnt = 0


def register_bound(net):
    global list_num
    for param in net.modules():
        if (isinstance(param, nn.ReLU)):
            param.register_forward_hook(bound_relu)
            #print('in register: ',list_num)

