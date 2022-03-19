

import utils
import torch
import numpy as np
import time
import os
import procedure

import config

import loss
import model
model_class = {
    'mymodel'       : model.MyModel,
    'flat_attention': model.Flat_Attention
}

print(f"dataset: {config.dataset_choice}")
print(f"model  : {config.model_choice}")
print(f"device : {config.info['device']}")

Recmodel = model_class[config.model_choice](
                **config.model_args[config.model_choice]
            )\
           .to(config.info['device'])
bpr = loss.BPRLoss(recmodel=Recmodel, **config.loss)

k = config.test['k']
best_valid_loss = float('inf')
log = open(config.info['weight_folder']+'log_'+f"{config.model_choice}_{config.dataset_choice}.txt", 'w')
acc_trace = utils.AccTrace()

for epoch in range(config.train['num_epoch']):
    print('======================')
    print(f'EPOCH[{epoch}]')
    start = time.time()

    if epoch % config.train['tr_per_te'] == 0:
        utils.cprint("[TEST]")
        result = procedure.Test(recmodel=Recmodel, epoch=epoch, **config.test) # {'prec': prec, 'rr':rr}
        torch.save(Recmodel.state_dict(), \
            config.info['weight_folder']+f"{config.model_choice}_{config.dataset_choice}_{epoch}.pth.tar")
        log.write(f"epoch:{epoch:5d} "+str(result)+'\n')

        output_information, valid_loss = procedure.BPR_valid(recmodel=Recmodel, loss_class=bpr, **config.train)
        print(f'{output_information}')
        if valid_loss < best_valid_loss:
            '''累计n次小于最优值就early stop'''
            acc_trace.fail_add_one()
            if acc_trace.count == config.train['num_earlyStop']:
                utils.cprint("[EARLY STOP]")
                print('*'*15)
                print("BEST EPOCH:\n", best_epoch)
                print('*'*15)
                break
        else:
            best_valid_loss = valid_loss
            acc_trace.reset()

    output_information = procedure.BPR_train(recmodel=Recmodel, loss_class=bpr, **config.train)
    print(f'{output_information}')
    
    '''
    print(f'[saved][{output_information}]')
    torch.save(Recmodel.state_dict(), config['weight_file'])
    print(f"[TOTAL TIME] {time.time() - start}")
    '''
log.close()