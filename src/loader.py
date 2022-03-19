
'''
input:
    train.txt: ([i1, i2, ...],[t1, t2, ...]) len = 13
    test.txt

add neg 

generate batch 
    users
    candidates

   users: namedtuple
       users.history: (batch_size, n_node)
       users.timestp
       users.adj    : (batch_size, n_node, n_node)
   candidates: namedtuple
       candidates.pos: (batch_size, 1)
       candidates.neg:(batch_size, 1)
'''

import numpy as np
import torch
from collections import namedtuple

import sys
sys.path.append('..')
import config

batch_size = config.train['batch_size']
train_size = config.train['history_size']
from collections import defaultdict 
one_time_check = defaultdict(lambda:1)

def neg_generator(one_sample:list, id_max = 11667):

    while True:
        neg_id = np.random.randint(1, id_max+1)
        if neg_id in one_sample:
            continue
        else:
            break
    return neg_id


def generate_batch(file, batch_size, device):
    '''
    file: './data/steam/train.txt'

    output: 
        users: namedtuple
           users.history: (batch_size, n_node)
           users.timestp: (batch_size, n_node)
           users.adj    : (batch_size, n_node, n_node)
        candidates: namedtuple
           candidates.pos: (batch_size, 1)
           candidates.neg: (batch_size, 1)
           candidates.stp: (batch_size, 1)
    '''

    n_item = config.dataset[config.dataset_choice]['num_items_all']

    f = open(file, 'r')
    while True:

        Users = namedtuple('User', ['history', 'timestp', 'adj'])
        Candidates = namedtuple('Candidates', ['pos', 'neg', 'stp'])

        history = torch.zeros((batch_size, train_size), dtype=torch.long).to(device)
        timestp = torch.zeros((batch_size, train_size), dtype=torch.long).to(device)

        _adj    = torch.ones((train_size, train_size)) - torch.eye(train_size)
        adj     = _adj.repeat(batch_size, 1).view((batch_size, train_size, train_size)).to(device)

        pos     = torch.zeros((batch_size, 1), dtype=torch.long).to(device)
        neg     = torch.zeros((batch_size, 1), dtype=torch.long).to(device)
        cdd_stp = torch.zeros((batch_size, 1), dtype=torch.long).to(device)

        len_batch = 0
        while(len_batch < batch_size):
            
            try:
                hst, stp = eval(f.readline())
                assert len(hst) == train_size + 1
            except:
                f = open(file, 'r')
                hst, stp = eval(f.readline())

            _stp = torch.tensor(stp)
            stp = (_stp - _stp[0])/(3600*24) + 1 # position for pad item is 0

            '''
            if one_time_check['stp']: 
                print(f'(one time check) stp:\n{stp}')
                one_time_check['stp'] = 0
            '''

            hst_train = torch.tensor(hst[:train_size])
            stp_train = stp[:train_size]

            pos_id = torch.tensor([hst[-1]])
            neg_id = torch.tensor([neg_generator(hst, n_item)])
            _cdd_stp = stp[-1:]

            history[len_batch] = hst_train
            timestp[len_batch] = stp_train
            pos[len_batch] = pos_id
            neg[len_batch] = neg_id
            cdd_stp[len_batch] = _cdd_stp

            len_batch += 1

            #print(history.dtype)

            #raise Exception('break')

        users = Users._make([history, timestp, adj])
        candidates = Candidates._make([pos, neg, cdd_stp])

        yield users, candidates

def generate_batch_test(path, batch_user, device):
    '''
    output: 
        users: namedtuple
           users.history: (batch_size, n_node)
           users.timestp
           users.adj    : (batch_size, n_node, n_node)
        candidates: namedtuple
           candidates.cdd: (batch_size, 1)
           candidates.stp:(batch_size, 1)
    '''
    test_file = path + '/test.txt'
    test_cdd_file = path + '/test_candidate_1_50.npy'

    test_all = open(test_file, 'r').readlines()
    test_cdd_all = np.load(test_cdd_file)

    for test, test_cdd in zip(test_all, test_cdd_all):

        test = eval(test)
        assert test[0][-1]==test_cdd[0]
        
        batch_size = len(test_cdd)

        _stp = torch.tensor(test[1])
        stp = (_stp - _stp[0])/(3600*24) + 1 # position for pad item is 0
        stp = stp.long()

        '''
        if one_time_check['stp']: 
                print(f'(one time check) stp:\n{stp}')
                one_time_check['stp'] = 0
        '''

        history = torch.tensor(test[0][:train_size], dtype=torch.long).repeat(batch_size, 1).to(device)
        timestp = stp[:train_size].repeat(batch_size, 1).to(device)
        _adj    = torch.ones((train_size, train_size)) - torch.eye(train_size)
        adj     = _adj.repeat(batch_size, 1).view((batch_size, train_size, train_size)).to(device)

        cdd     = torch.tensor(test_cdd).view((batch_size, 1)).to(device)
        cdd_stp = stp[-1:].repeat(batch_size, 1).to(device)
        label   = [1]*1 + [0]*50

        Users = namedtuple('User', ['history', 'timestp', 'adj'])
        Candidates = namedtuple('Candidates', ['cdd', 'stp', 'label'])

        users = Users._make([history, timestp, adj])
        candidates = Candidates._make([cdd, cdd_stp, label])

        yield users, candidates


if __name__=='__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    '''
    u, c = next(generate_batch('./data/steam/train.txt', batch_size = 2, device = device))

    print(u.history.dtype)

    print(c)
    '''
    u, c = next(generate_batch_test('steam', path='./',device = device))