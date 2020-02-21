import torch.nn as nn
import torch.nn.functional as F
from GCN.GraphConvolution import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid_list, nclass, dropout):
        super(GCN, self).__init__()
        
        self.nfeat = nfeat
        self.nhid_list = nhid_list
        self.nclass = nclass
        self.dropout = dropout
        self.gc_list = nn.ModuleList()
        self.model_layer_setup()
        return
    
    def model_layer_setup(self):
        nfeat_list = [self.nfeat] + self.nhid_list
        for i in range(1,len(nfeat_list)):
            self.gc_list.append(GraphConvolution(nfeat_list[i-1], nfeat_list[i]))
        self.gc_list.append(GraphConvolution(nfeat_list[len(nfeat_list)-1], self.nclass))
        return

    def forward(self, x, adj):
        for i in range(len(self.nhid_list)):
            x = F.relu(self.gc_list[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc_list[len(self.nhid_list)](x, adj)
        return F.log_softmax(x, dim=1)