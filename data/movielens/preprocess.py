
'''
ratings.dat
    UserID::MovieID::Rating::Timestamp
movies.dat
    MovieID::Title::Genres
'''

'''
target file:
    train.txt: 
        ith row: ([itm1, itm2, ...],[stp1, stp2, ...]) id start from 1
    test.txt
    test_candidate_1_50.npy
'''

'''
user2asin:{user:([itm1, ...], [stp1,...]}
'''
import pickle
from tqdm import tqdm

data = {}

with open('ratings.dat', 'rb') as f:

    for row in tqdm(f.readlines()):

        uid, iid, rating, stp = str(row, encoding='utf8').strip().split('::')

        if uid not in data:
            data[uid] = ([],[])
        data[uid][0].append(iid)
        data[uid][1].append(stp)

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

######

import numpy as np

len_sample = 13

all_sample = {}
'''
all_sample[sample_id] = ([itm],[stp]) 
sample_id starts from 0
'''
sample_id = 0
for u in tqdm(data):
    l = len(data[u][0])
    if l < len_sample:
        continue

    ind = np.argsort(np.array(data[u][1]))
    itm_list = list(np.array(data[u][0])[ind])
    stp_list = list(np.array(data[u][1])[ind])

    for start in range(l//len_sample):

        all_sample[sample_id] = (itm_list[start:start+len_sample] , stp_list[start:start+len_sample])
        sample_id += 1

with open('all_sample.pkl', 'wb') as f:
    pickle.dump(all_sample, f, pickle.HIGHEST_PROTOCOL)

print(f'all_sample: {len(all_sample)}')

######

# split train test 10:1
import numpy as np
np.random.seed(63)

import pickle
from tqdm import tqdm

with open('all_sample.pkl', 'rb') as f:
    all_sample = pickle.load(f)

#编号 start from 1
itm2id,id2itm = {},{}
for u in all_sample:
    for itm in all_sample[u][0]:
        if itm not in itm2id:
            id2itm[len(itm2id)+1] = itm
            itm2id[itm] = len(itm2id)+1
with open('id2itm.pkl', 'wb') as f:
    pickle.dump(id2itm, f, pickle.HIGHEST_PROTOCOL)
with open('itm2id.pkl', 'wb') as f:
    pickle.dump(itm2id, f, pickle.HIGHEST_PROTOCOL)

n_sample = len(all_sample)
idx_all = np.arange(n_sample)
np.random.shuffle(idx_all)

f_train = open('train.txt', 'w')
f_valid = open('valid.txt', 'w')
f_test = open('test.txt', 'w')

str_train, str_valid, str_test = '', '', ''

for i in tqdm(idx_all[:n_sample//10]): 
    itm_list = [itm2id[itm] for itm in all_sample[i][0]]
    stp_list = all_sample[i][1]
    str_test += str((itm_list, stp_list)) + '\n'
f_test.write(str_test)
f_test.close()

n_valid = (n_sample - n_sample//10)//10
for i in tqdm(idx_all[n_sample//10:n_sample//10+n_valid]): 
    itm_list = [itm2id[itm] for itm in all_sample[i][0]]
    stp_list = all_sample[i][1]
    str_valid += str((itm_list, stp_list)) + '\n'
f_train.write(str_valid)
f_train.close()


for i in tqdm(idx_all[n_sample//10+n_valid:]): 
    itm_list = [itm2id[itm] for itm in all_sample[i][0]]
    stp_list = all_sample[i][1]
    str_train += str((itm_list, stp_list)) + '\n'
f_train.write(str_train)
f_train.close()

#####

import numpy as np

np.random.seed(63)

#n_item = 3390
import pickle
id2itm = pickle.load(open('id2itm.pkl', 'rb'))
n_item = max(id2itm.keys())
n_test_sample = 50
test_all = []

with open('./test.txt', 'r') as f:

    for row in f.readlines():
        row = eval(row)

        test_sample = []
        test_sample.append(row[0][-1])

        while len(test_sample) < n_test_sample+1:
            
            cdd = np.random.randint(1, n_item+1)
            if cdd in row[0]:
                continue
            test_sample.append(cdd)

        test_all.append(test_sample)

np.save('./test_candidate_1_50.npy', np.array(test_all))





