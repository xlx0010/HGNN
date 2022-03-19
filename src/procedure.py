
import multiprocessing
import utils
from tqdm import tqdm
import numpy as np
import torch

from loader import generate_batch, generate_batch_test
from metrics import compute_metrics, compute_diversity
import config

CORES = multiprocessing.cpu_count() // 2

def BPR_train(path, recmodel, loss_class, batch_size, device, **kwargs):
    Recmodel = recmodel
    Recmodel.train()
    bpr: BPRLoss = loss_class

    with open(path + '/train.txt', 'r') as f:
        train_size = len(f.readlines())

    total_batch = train_size//batch_size + 1
    aver_loss = 0.
    
    batch_generator = generate_batch(file=path + '/train.txt', batch_size=batch_size, device = device)

    for batch_i in tqdm(range(total_batch), total=float("inf")):
        batch_users, batch_candidates = next(batch_generator)
        cri = bpr.stageOne(batch_users, batch_candidates)
        aver_loss += cri
        
    aver_loss = aver_loss / total_batch
    return f"[BPR train [aver loss{aver_loss:.3e}]"

def BPR_valid(path, recmodel, loss_class, batch_size, device, **kwargs):
    Recmodel = recmodel
    Recmodel.eval()
    bpr: BPRLoss = loss_class

    with open(path + '/valid.txt', 'r') as f:
        valid_size = len(f.readlines())

    total_batch = valid_size//batch_size + 1
    aver_loss = 0.
    
    batch_generator = generate_batch(file=path + '/valid.txt', batch_size=batch_size, device = device)

    for batch_i in tqdm(range(total_batch), total=float("inf")):
        batch_users, batch_candidates = next(batch_generator)
        cri = bpr.stageEval(batch_users, batch_candidates)
        aver_loss += cri
        
    aver_loss = aver_loss / total_batch
    return f"[BPR valid [aver loss{aver_loss:.3e}]", valid_loss

def Test(path, recmodel, batch_user, device, epoch, k=20, **kwargs):
    
    Recmodel = recmodel
    Recmodel.eval()
    
    batch_generator_test = generate_batch_test(path=path, batch_user = batch_user, device = device)

    with torch.no_grad():
    
        accs_all = []
        rec_result_all = []

        for user, candidate in tqdm(batch_generator_test):

            scores = Recmodel.compute_rating(user, candidate).cpu().numpy()

            label_origin = candidate.label
            accs = compute_metrics(scores, label_origin, k)
            
            divsty = compute_diversity(scores, candidate.cdd, Recmodel, k)
            accs.append(divsty)

            accs_all.append(accs)

            cdd_sorted = candidate.cdd.squeeze().cpu().numpy()[np.argsort(scores)[::-1]]
            rec_result_all.append(cdd_sorted)

        aver_accs = np.mean(np.array(accs_all), axis = 0)
        hit, mrr, div = aver_accs
    
    rec_result_all = np.array(rec_result_all)
    np.save(config.info['weight_folder']+f"{epoch}_recresult_{config.model_choice}_{config.dataset_choice}.npy", rec_result_all)

    result = f"{{\'hit@{k}\':{hit:.3e}, \'mrr@{k}\':{mrr:.3e},  \'div@{k}\':{div:.3e}}}"
    print(result)


    return eval(result)

if __name__=='__main__':

    from model import MyModel
    import loss

    config = {
        'batch_size': 1024,
        'decay': 1e-4,
        'lr': 1e-3
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Recmodel = MyModel(dim_node=12,\
                dim_hidden=64, \
                n_assign=5, \
                n_heads=2, \
                num_items_all=11667 \
                ).to(device)
    #Recmodel = Recmodel.to(device)

    '''
    bpr = loss.BPRLoss(Recmodel, config)

    output_information = BPR_train('./data/steam', Recmodel, bpr, 1024)
    '''

    print(Test(path='./data/steam', recmodel=Recmodel, device=device))


