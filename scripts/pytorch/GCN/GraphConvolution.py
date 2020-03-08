# Authors: Aaron Wu / Howard Tai

# Script defining a GraphConvolution PyTorch module including forward pass and weight initializations

import pdb
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, init='kaiming'):
        """
        Input(s):
        - in_features (int): Number of input features
        - out_features (int): Number of output features
        - init (string): Specifies weight initialization strategy
        """
        super(GraphConvolution, self).__init__()

        self.n_in = in_features
        self.n_out = out_features

        self.weight = Parameter(torch.zeros(size=(self.n_in, self.n_out), dtype=torch.float32), requires_grad=True)
        self.bias = Parameter(torch.zeros(size=(self.n_out,), dtype=torch.float32), requires_grad=True)

        # Set initializations
        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()

        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()

        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()

    def reset_parameters_uniform(self):
        """
        Resets parameters back to Uniform initialization
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        """
        Resets parameters back to Xavier initialization
        """
        torch.nn.init.xavier_normal_(self.weight.data, gain=0.02)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        """
        Resets parameters back to Kaiming initialization, also known as He initialization
        """
        torch.nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):
        """
        Defines forward pass for graph convolution
        """
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.n_in) + ' -> ' \
               + str(self.n_out) + ')'
