
import config

def module_test_print(var_input, var_inmed, var_ouput):
    for var in (var_input, var_inmed, var_ouput):
        print('*'*10)
        for key, value in var.items():
            print('#', key)
            print(value)

def UniformSample(users, dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    return:
        np.array
    """
    dataset : BasicDataset
    users = np.random.randint(0, dataset.n_users, dataset.trainDataSize)
    History = dataset.history
    S = []
    for i, user in enumerate(users):
        history = History[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

class AccTrace():

    def __init__(self):
        self.count = 0
    
    def fail_add_one(self):
        self.count += 1

    def reset(self):
        self.count = 0
    
def get_weight_file(epoch):

    return config.info['weight_folder']+\
        f"{config.model_choice}_{config.dataset_choice}_{epoch}.pth.tar"