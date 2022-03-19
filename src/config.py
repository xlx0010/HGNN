
import torch

#----------------------------------------------------------------------
info = {
    'device'       : torch.device('cuda:0' if torch.cuda.is_available else 'cpu'),
    'weight_folder': './weights/',
    'cwd'          : '.'
}
info['device'] = torch.device('cpu')

#----------------------------------------------------------------------
dataset_choice = 'movielens'
# steam, movielens, amazon_movie
dataset = {
    'steam':{ 
        'path'             : info['cwd']+'data/steam/',
        'onesample_size'   : 12,
        'num_items_all'    : 11667,
        'max_position_span': None 
    },
    'movielens':{ 
        'path'             : info['cwd']+'data/movielens/',
        'onesample_size'   : 12,
        'num_items_all'    : 3390,
        'max_position_span': None 
    },
    'amazon_movie':{ 
        'path'             : info['cwd']+'data/amazon_movie/',
        'onesample_size'   : 12,
        'num_items_all'    : 81690,
        'max_position_span': None 
    },
    'amazon_book':{ 
        'path'             : info['cwd']+'data/amazon_book/',
        'onesample_size'   : 100,
        'num_items_all'    : 677847,
        'max_position_span': None 
    }
}

#----------------------------------------------------------------------
model_choice = 'mymodel'
# mymodel, flat_attention
model_args = {
    'mymodel':{
        'dim_node'     : dataset[dataset_choice]['onesample_size'],
        'num_items_all': dataset[dataset_choice]['num_items_all'],
        'dim_hidden'   : 64,
        'n_assign'     : 5,
        'n_heads'      : 2,
        'device'       : info['device']
    },
    'flat_attention':{
        'dim_node'     : dataset[dataset_choice]['onesample_size'],
        'num_items_all': dataset[dataset_choice]['num_items_all']
    }
}

loss = {
    'decay_reg': 1e-4,
    'decay_ent': 1e-4,
    'lr'   : 1e-3
}

train = {
    'path'         : info['cwd']+f'data/{dataset_choice}',
    'num_epoch'    : 10000,
    'history_size' : dataset[dataset_choice]['onesample_size'],
    'batch_size'   : 1024,
    'device'       : info['device'],
    'tr_per_te'    : 1,
    'num_earlyStop': 20
}

test = {
    'path'         : info['cwd']+f'data/{dataset_choice}',
    'batch_user'   : 1,
    'device'       : info['device'],
    'k'            : 20
}

#-------------------------------script---------------------------------

# dataset['max_position_span']
max_position_span = -float('inf')
for filename in ['train.txt', 'test.txt']:
    with open(dataset[dataset_choice]['path'] + filename, 'r') as f:
        for row in f:
            row = eval(row)[1]  
            span = (row[-1]-row[0])/(3600*24)
            max_position_span = max(span, max_position_span)
dataset[dataset_choice]['max_position_span'] = int(max_position_span+1)

# weight_folder
# clear empty folder
import os
for item in os.listdir(info['weight_folder']):
    item_path = os.path.join(info['weight_folder'], item)
    if os.path.isdir(item_path) and len(os.listdir(item_path))==0:
        os.rmdir(item_path)
# prepare weight_folder
from datetime import datetime
info['weight_folder'] += datetime.now().strftime('%m%d%H%M')+'/'
if not os.path.exists(info['weight_folder']):
    os.mkdir(info['weight_folder'])

