

import gzip
import numpy as np
from collections import defaultdict
from datetime import datetime
import pickle
from tqdm import tqdm

'''
>>> rev.keys()
dict_keys(['username', 'hours', 'products', 'product_id', 'page_order', 'date', 'text', 'early_access', 'page'])
    {
        'username': 'Chaos Syren',
        'product_id': '725280',
        'date': '2017-12-17'
    }
'''

class DataPreprocessor:

    def __init__(self):
        super(DataPreprocessor, self).__init__()

        self.usr2id, self.itm2id = {}, {}

        self.DATA = defaultdict(dict)
        '''{
            usr_id:
                {
                    history  : [id_0, id_1, ...]
                    timestmp: [tm_0, tm_1, ...]
                }
        }'''

    def get_id(self, obj, table):
        if obj in table:
            return table[obj]
        else:
            i = len(table) + 1
            table[obj] = i
            return i

    def date2stmp(self, date, fmt):
        # '%Y-%m-%d %H:%M:%S'
        return datetime.strptime(date, fmt).timestamp()
    
    def sortList1ByList2(self, list1, list2):
        '''
        ascending order
        return numpy.ndarray
        '''
        if type(list1)==list:
            list1, list2 = np.array(list1), np.array(list2)
        ind = np.argsort(list2)
        return list1[ind], list2[ind]

    def save_DATA_into(self, filename):
        print('saving DATA into {}...'.format(filename))
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.DATA, output, pickle.HIGHEST_PROTOCOL)

    def run_review(self, file_name):
        
        f = gzip.open(file_name, 'r')
        print('processing {}...'.format(file_name))
        for row in tqdm(f):
            row = eval(row)
            _usr, _itm, _dat = row['username'], row['product_id'], row['date']

            usr, itm = self.get_id(_usr, self.usr2id), self.get_id(_itm, self.itm2id)
            dat = _dat#dat = self.date2stmp(_dat, '%Y-%m-%d')

if __name__=='__main__':

    import os
    data_pre_file = 'data_pre.pkl'

    data_pre = DataPreprocessor()
    #print(data_pre.sortList1ByList2([1,2,3], [1,3,2]))
    data_pre.run_review('steam_reviews.json.gz')
    #data_pre.save_DATA_into(data_pre_file)
    print('dumping')
    with open('usr2id.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(data_pre.usr2id, output, pickle.HIGHEST_PROTOCOL)
    with open('itm2id.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(data_pre.itm2id, output, pickle.HIGHEST_PROTOCOL)
    #####

    def date2stmp(date, fmt):
        # '%Y-%m-%d %H:%M:%S'
        return datetime.strptime(date, fmt).timestamp()

    with open('data_pre.pkl', 'rb') as f:
        data = pickle.load(f)

    one_sample_size = 13 # 12 + 1

    f_hst = open('history.txt', 'w')
    f_stp = open('timestamp.txt', 'w')

    all_hst, all_stp = '', ''

    for usr in tqdm(data):

        if not len(data[usr]['history']) >= one_sample_size: continue

        _history = [str(i) for i in list(data[usr]['history'])]
        line_hst = ' '.join(_history) + '\n'
        all_hst += line_hst

        _timestp = [str(date2stmp(t, '%Y-%m-%d')) for t in list(data[usr]['timestmp'])]
        line_stp = ' '.join(_timestp) + '\n'
        all_stp += line_stp

    f_hst.write(all_hst)
    f_stp.write(all_stp) 

    f_hst.close()
    f_stp.close()

    #####

    f_hst = open('history.txt', 'r')
    f_stp = open('timestamp.txt', 'r')

    all_sample = {}
    one_sample_size = 13 # 12 + 1
    id_map = {}
    n_sample = 1

    def get_id(obj, table):
        if obj in table:
            return table[obj]
        else:
            i = len(table) + 1 # start with 1
            table[obj] = i
            return i

    row_hst = f_hst.readline().strip().split()
    row_stp = f_stp.readline().strip().split()

    while row_hst:

        for start in range(len(row_hst)//one_sample_size):

            smp_hst = [get_id(itm, id_map) for itm in row_hst[start:start+one_sample_size]]

            smp_stp = [eval(stp) for stp in row_stp[start:start+one_sample_size]]
            all_sample[n_sample] = (smp_hst, smp_stp)

            n_sample += 1


        row_hst = f_hst.readline().strip().split()
        row_stp = f_stp.readline().strip().split()

    f_hst.close()
    f_stp.close()

    print('dumping id_map to itm2id_2.pkl')
    import pickle
    with open('itm2id_2.pkl', 'wb') as f:
        pickle.dump(id_map, f, pickle.HIGHEST_PROTOCOL)

    #####

    np.random.seed(63)

    smp_idx = np.arange(1, max(all_sample.keys())+1)
    np.random.shuffle(smp_idx)

    print(smp_idx[:10])

    f_train = open('train.txt', 'w')
    f_valid = open('valid.txt', 'w')
    f_test = open('test.txt', 'w')

    str_train, str_valid, str_test = '', '', ''

    for i in tqdm(smp_idx[:n_sample//10]): 
        test_smp = all_sample[i]
        str_test += str(test_smp) + '\n'
    f_test.write(str_test)
    f_test.close()

    n_valid = (n-n_sample//10)//10
    for i in tqdm(smp_idx[n_sample//10:n_sample//10+n_valid]): 
        valid_smp = all_sample[i]
        str_valid += str(valid_smp) + '\n'
    f_valid.write(str_valid)
    f_valid.close()

    for i in tqdm(smp_idx[n_sample//10+n_valid:]): 
        train_smp = all_sample[i]
        str_train += str(train_smp) + '\n'
    f_train.write(str_train)
    f_train.close()

    # n_item
    print('n_item = {}'.format(max(id_map.values())))

    #####

    n_item = 11667
    n_test_sample = 50
    test_all = []

    with open('./data/steam/test.txt', 'r') as f:

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

    np.save('./data/steam/test_candidate_1_50.npy', np.array(test_all))

