# -*- coding: utf-8 -*-
import numpy as np
from operator import itemgetter

def hit_score(y_true, k=5):
    hit = float(sum(y_true[:k]))
    return hit

'''
def ndcg_score(y_true, k=5):
    idcg = float(sum([1 / np.log2(i + 1) for i in np.arange(1, k + 1)]))
    dcg = float(sum((2 ** np.array(y_true)[:k] - 1) / np.log2(np.arange(1, k + 1) + 1)))
    return float(dcg / idcg)

def ap_score(y_true, k=5):
    p = float(0)
    for i in range(0, k):
        if (y_true[i] == 1):
            p += float(sum(y_true[:(i + 1)]) / (i + 1))
    ap = float(p / k)
    return ap
'''
def rr_score(y_true, k):
    idx = y_true.index(1) + 1
    return 1.0 / idx if idx<=k else 0

def compute_metrics(score, label, k):
    '''
    score: [score1, score2, ...] np.array
    label: [label1, label2, ...] list
    '''
    sort_idx = np.argsort(score)[::-1]
    y_true = list(np.array(label)[sort_idx])

    return [
        hit_score(y_true, k),\
        rr_score(y_true, k)
        ]

import torch 
import config
def compute_diversity(score, cdd, recmodel, k):
    '''
    scores: (batch_size) np.array
    cdd: (batch_size, 1) torch.tensor
    '''
    recmodel.eval()
    with torch.no_grad():

        sort_idx = np.argsort(score)[::-1]
        cdd_k = cdd.squeeze().cpu().numpy()[sort_idx[:k]]
        cdd_k = torch.from_numpy(cdd_k).to(config.info['device'])
        # (k,) np.array
        C = recmodel.embedding_item(cdd_k) # (k, dim_node)
        C_t = torch.transpose(C, 0, 1)
        ILD = (2.0/k*(k-1))*torch.sum(torch.matmul(C, C_t))

    return float(ILD)

if __name__=='__main__':

    y_true = [1,0,0,1,0]

    print(prec_score(y_true))
    print(ndcg_score(y_true))
    print(ap_score(y_true))
    print(rr_score(y_true))

    print(compute_metrics([1,2,3,4,5], [1,0,0,0,0]))
