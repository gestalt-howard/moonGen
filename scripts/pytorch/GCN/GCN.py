# Authors: Aaron Wu / Howard Tai

# Script for defining GCN model structure and forward propagation

import torch.nn as nn
import torch.nn.functional as F
from pytorch.GCN.GraphConvolution import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeatures, nhidden_layer_list, nclass, dropout):
        """
        Input(s):
        - nfeatures (int)
        - nhidden_layer_list (list of int)
        - nclass (int)
        - dropout (float)
        """
        super(GCN, self).__init__()

        self.nfeat = nfeatures
        self.nhid_list = nhidden_layer_list
        self.nclass = nclass
        self.dropout = dropout
        self.gc_list = nn.ModuleList()
        self.model_layer_setup()

    def model_layer_setup(self):
        """
        Defines layers in GCN model
        """
        nfeat_list = [self.nfeat] + self.nhid_list  # Input features + hidden features

        # Define intermediate layers
        for i in range(1, len(nfeat_list)):
            self.gc_list.append(GraphConvolution(nfeat_list[i - 1], nfeat_list[i]))

        # Define output layer
        self.gc_list.append(GraphConvolution(nfeat_list[len(nfeat_list) - 1], self.nclass))
        return None

    def forward(self, x, adj):
        """
        Input(s):
        - x (PyTorch tensor): Input features
        - adj (PyTorch tensor): Adjacency matrix
        """
        # Forward pass is Graph Convolution -> ReLU -> Dropout
        for i in range(len(self.nhid_list)):
            x = F.relu(self.gc_list[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)

        # Get logits
        x = self.gc_list[len(self.nhid_list)](x, adj)
        return F.log_softmax(x, dim=1)
