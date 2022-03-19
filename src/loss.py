
import torch
from torch import nn, optim
import config

class BPRLoss:
    def __init__(self, 
                 recmodel, 
                 decay_reg, decay_ent, lr, **kwargs):
        self.model = recmodel
        self.decay_reg = decay_reg
        self.decay_ent = decay_ent
        self.lr = lr
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        
    def stageOne(self, users, candidates):
                    
        loss, reg_loss, entropy_loss = self.model.bpr_loss(users, candidates)
        reg_loss = reg_loss*self.decay_reg
        entropy_loss = entropy_loss*self.decay_ent
        loss = loss + reg_loss + entropy_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.cpu().item()
    
    def stageEval(self, users, candidates):
                    
        loss, reg_loss, entropy_loss = self.model.bpr_loss(users, candidates)
        reg_loss = reg_loss*self.decay_reg
        entropy_loss = entropy_loss*self.decay_ent
        loss = loss + reg_loss + entropy_loss
        
        return loss.cpu().item()
