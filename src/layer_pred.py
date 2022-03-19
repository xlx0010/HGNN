import torch
import torch.nn as nn
import torch.nn.functional as F

class PredLayer(nn.Module):
    def __init__(self, module_test=False):
        super(PredLayer, self).__init__()
        self.module_test = module_test

    def forward(self, u, v):
        '''
        input:
            u: (batch_size, 1, dim_user)
            v: (batch_size, 1, dim_item)
        output:
            y: (batch_size, 1)   
        '''
        _y = torch.mul(u, v) # (batch_size, 1, dim_user)
        y = torch.sum(_y, dim=-1) # (batch_size, 1)

        if self.module_test:
            var_input = ['u', 'v']
            var_inmed = ['_y']
            var_ouput = ['y']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )

        return y 

if __name__=='__main__':
    
    from utils import module_test_print

    Module_Test = PredLayer(module_test=True)
    u = torch.arange(4, dtype = torch.float32).view((2,1,2))
    v = torch.arange(4, dtype = torch.float32).view((2,1,2))
        
    output = Module_Test(u, v)
