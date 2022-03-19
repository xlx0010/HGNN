import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_gat import MultiHeadsAttentionLayer

class ParentNodeLayer(nn.Module):

    def __init__(self, dim_node, dim_hidden, n_heads, n_assign, device,\
                 module_test = False):

        super(ParentNodeLayer, self).__init__()

        self.GAT_embed  = MultiHeadsAttentionLayer(dim_node, dim_hidden, dim_node, n_heads, device).to(device)
        self.GAT_assign = MultiHeadsAttentionLayer(dim_node, dim_hidden, n_assign, n_heads, device).to(device)

        self.module_test = module_test

    def forward(self, x, adj, t):
        '''
        input: 
            x  : (batch_size, n_node, dim_node)
            adj: (batch_size, n_node, n_node)
            t  : (batch_size, 1, n_node)
        output: 
            x_new: (batch_size, assign, dim_node)
        '''

        A = torch.zeros_like(adj)
        batch_size, n_node, _ = adj.size()
        
        for i in range(batch_size):
            for j in range(n_node):
                for k in range(n_node):
                    A[i,j,k] = 1/abs(t[i,0,j]-t[i,0,k])

        z = self.GAT_embed(x, adj, A) # (batch_size, n_node, dim_node)
        s = nn.Softmax(dim=-1)(self.GAT_assign(z, adj, A)) # (batch_size, n_node, n_assign)
        x_parent = torch.matmul(torch.transpose(s, -2, -1), z) # (batch_size, n_assign, dim_node)

        t_parent = torch.bmm(t, s)

        if self.module_test:
            var_input = ['x', 'adj']
            var_inmed = ['z', 's']
            var_ouput = ['x_parent']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )

        return x_parent, s, t_parent

    def forward_casestudy(self, x, adj):
        '''
        input: 
            x  : (batch_size, n_node, dim_node)
            adj: (n_node, n_node)
        output: 
            x_new: (batch_size, assign, dim_node)
        '''
        z = self.GAT_embed(x, adj) # (batch_size, n_node, dim_node)
        s = nn.Softmax(dim=-1)(self.GAT_assign(z, adj)) # (batch_size, n_node, n_assign)
        x_parent = torch.matmul(torch.transpose(s, -2, -1), z) # (batch_size, n_assign, dim_node)

        if self.module_test:
            var_input = ['x', 'adj']
            var_inmed = ['z', 's']
            var_ouput = ['x_parent']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )

        return x_parent, s, z
        

if __name__=='__main__':

    from utils import module_test_print

    Module_Test = ParentNodeLayer(dim_node=3, dim_hidden=3, n_assign=2, n_heads=2, module_test=True)
    x = torch.tensor([[1.,2.,3.],[3.,4.,5.]]).repeat(2,1).view((2,2,3))
    adj = torch.tensor([[1.,1.],[0.,1.]]).repeat(2,1).view((2,2,2))
        
    output = Module_Test(x, adj)