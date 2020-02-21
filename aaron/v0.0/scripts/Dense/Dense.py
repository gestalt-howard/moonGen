import torch.nn as nn
import torch.nn.functional as F
from Dense.DenseLayer import DenseLayer

class Dense(nn.Module):
    def __init__(self, nfeat, nhid_list, nclass, dropout):
        super(Dense, self).__init__()
        
        self.nfeat = nfeat
        self.nhid_list = nhid_list
        self.nclass = nclass
        self.dropout = dropout
        self.hidden_list = nn.ModuleList()
        self.model_layer_setup()
        return
    
    def model_layer_setup(self):
        nfeat_list = [self.nfeat] + self.nhid_list
        for i in range(1,len(nfeat_list)):
            self.hidden_list.append(DenseLayer(nfeat_list[i-1], nfeat_list[i]))
        self.hidden_list.append(DenseLayer(nfeat_list[len(nfeat_list)-1], self.nclass))
        return

    def forward(self, x):
        for i in range(len(self.nhid_list)):
            x = F.relu(self.hidden_list[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.hidden_list[len(self.nhid_list)](x)
        return F.log_softmax(x, dim=1)