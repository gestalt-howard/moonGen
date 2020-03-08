# Authors: Aaron Wu / Howard Tai

# Script for defining Dense model structure and forward propagation

import torch.nn as nn
import torch.nn.functional as F
from pytorch.Dense.DenseLayer import DenseLayer


class Dense(nn.Module):
    def __init__(self, nfeatures, nhidden_layer_list, nclass, dropout):
        """
        Input(s):
        - nfeatures (int)
        - nhidden_layer_list (list of int)
        - nclass (int)
        - dropout (float)
        """
        super(Dense, self).__init__()

        self.nfeat = nfeatures
        self.nhid_list = nhidden_layer_list
        self.nclass = nclass
        self.dropout = dropout
        self.hidden_list = nn.ModuleList()
        self.model_layer_setup()

    def model_layer_setup(self):
        """
        Defines layers in Dense model
        """
        nfeat_list = [self.nfeat] + self.nhid_list  # Input features + hidden features

        # Define intermediate layers
        for i in range(1, len(nfeat_list)):
            self.hidden_list.append(DenseLayer(nfeat_list[i - 1], nfeat_list[i]))

        # Define output layer
        self.hidden_list.append(DenseLayer(nfeat_list[len(nfeat_list) - 1], self.nclass))
        return

    def forward(self, x, adj):
        """
        Input(s):
        - x (PyTorch tensor): Input features
        - adj (PyTorch tensor): Adjacency matrix (not used in this forward pass)
        """
        # Forward pass is Fully-Connected -> ReLU -> Dropout
        for i in range(len(self.nhid_list)):
            x = F.relu(self.hidden_list[i](x))
            x = F.dropout(x, self.dropout, training=self.training)

        # Get logits
        x = self.hidden_list[len(self.nhid_list)](x)
        return F.log_softmax(x, dim=1)
