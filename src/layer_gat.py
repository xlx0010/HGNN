import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_att import GraphAttentionLayer

class MultiHeadsAttentionLayer(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, n_heads, device,\
                 dropout=0.0,\
                 module_test = False):

        super(MultiHeadsAttentionLayer, self).__init__()

        self.dropout = dropout

        self.atts_first = [GraphAttentionLayer(dim_input, dim_hidden, dropout).to(device) for _ in range(n_heads)]

        '''
        for i, att in enumerate(self.atts_first):
            self.add_module('attention_{}'.format(i), att)
        add_module有啥用
        '''

        self.att_last = GraphAttentionLayer(dim_hidden * n_heads, dim_output, dropout=dropout).to(device)

        self.module_test = module_test
    
    def forward(self, x, adj, A):

        x = F.dropout(x, self.dropout, training=self.training)
        x_multi = torch.cat([att(x, adj, A) for att in self.atts_first], dim=-1)
        x_multi = F.dropout(x_multi, self.dropout, training=self.training)
        x_new = self.att_last(x_multi, adj, A)

        if self.module_test:
            var_input = ['x', 'adj']
            var_inmed = ['x_multi']
            var_ouput = ['x_new']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )

        #return F.log_softmax(x_new, dim=1)
        return x_new

if __name__=='__main__':

    from utils import module_test_print

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Module_Test = MultiHeadsAttentionLayer(dim_input=2, dim_hidden=2, dim_output=2, n_heads=2, module_test=True).to(device)
    x = torch.tensor([[1.,2.],[3.,4.]]).repeat(2,1).view((2,2,2)).to(device)
    adj = torch.tensor([[1.,1.],[0.,1.]]).repeat(2,1).view((2,2,2)).to(device)
        
    output = Module_Test(x, adj)
