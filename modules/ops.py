import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import functools
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _routing_two(nn.Module):
    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing_two, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)
        self.fc2 = nn.Linear(in_channels, num_experts)

    def forward(self, x1, x2): 
        x1 = torch.flatten(x1)
        x1 = self.dropout(x1)
        x1 = self.fc(x1) #384->4

        x2 = torch.flatten(x2)
        x2 = self.dropout(x2)
        x2 = self.fc2(x2) #384->4
        return torch.sigmoid(x1 + x2)



def model_structure(model):
    blank = ' '
    # print('-' * 90)
    # print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|'
    #       + ' ' * 15 + 'weight shape' + ' ' * 15 + '|'
    #         + ' ' * 3 + 'number' + ' ' * 3 + '|')
    # print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print(
        'The parameters of Model {}: {:4f}M'.format(
            model._get_name(), num_para * type_size / 1000 / 1000
        )
    )
    print('-' * 90)
