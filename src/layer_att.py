# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:49:42 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# GAT basic operation
class GraphAttentionLayer(nn.Module):
    def __init__(self, dim_input, dim_output, \
                 dropout=0.0, negative_slope_LeakyRelu = 0.01, \
                 module_test = False):
        super(GraphAttentionLayer, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = dropout
        
        self.W = nn.Parameter(torch.empty(size=(dim_input, dim_output)))
        nn.init.xavier_uniform_(self.W.data, gain=nn.init.calculate_gain('leaky_relu', negative_slope_LeakyRelu))
        
        self.leakyrelu = nn.LeakyReLU(negative_slope_LeakyRelu)
        
        self.module_test = module_test
        
    def _attention_score(self, x, y=None):
        
        return torch.matmul(x, x.transpose(-2, -1)) # (n_node, n_node)
    
    def forward(self, x, adj, A):
        '''
        input:
            x   : (batch_size, n_node, dim_node)
            adj : (batch_size, n_node, n_node)
        '''
        
        xW = torch.matmul(x, self.W)# (batch_size, n_node, dim_input) * (dim_input, dim_output)
        #xW = torch.matmul(x, torch.ones((2,2))) # for module_test

        score = self._attention_score(xW) # (batch_size, n_node, n_node)
        score += A # add timespan weight
        
        zero_vec = -9e15*torch.ones_like(score)
        _attention = torch.where(adj > 0, score, zero_vec)

        attention = F.softmax(_attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        x_new = F.elu(torch.matmul(attention, xW))# (batch_size, N, dim_output)

        if self.module_test:
            var_input = ['x', 'adj']
            var_inmed = ['xW', 'score', 'zero_vec', '_attention', 'attention']
            var_ouput = ['x_new']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )
        
        return x_new

                
    def __repr__(self):
        return '{}({}->{})'.format(self.__class__.__name__, self.dim_input, self.dim_output)


if __name__=='__main__':

    from utils import module_test_print

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Module_Test = GraphAttentionLayer(dim_input=2, dim_output=2, module_test=True)

    x = torch.tensor([[1.,2.],[3.,4.]]).repeat(2,1).view((2,2,2)).to(device)
    adj = torch.tensor([[1.,1.],[0.,1.]]).repeat(2,1).view((2,2,2)).to(device)
        
    output = Module_Test(x, adj)






        