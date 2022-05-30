import collections
import random

import torch
import pickle
import os

#Hyperparameters
buffer_limit  = 30000
batch_size    = 512
bf_PATH = './buffer/'

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit*5)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, cr_lst, s_prime_lst, done_mask_lst, gr_lst = [], [], [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, cr, s_prime, done_mask, gr = transition
            s = torch.squeeze(s)
            s_prime = torch.squeeze(s)
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            cr_lst.append([cr])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            gr_lst.append([gr])

        return torch.stack(s_lst, dim=0), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(cr_lst), \
               torch.stack(s_prime_lst, dim=0), torch.tensor(done_mask_lst), torch.tensor(gr_lst)
    
    def size(self):
        return len(self.buffer)
        
    def save(self, start):
        with open(bf_PATH+str(start)+'.pickle', 'wb') as fw:
            pickle.dump(self.buffer, fw)
        
    def load(self):
        for bf in os.listdir(bf_PATH):
            with open(bf_PATH+bf, 'rb') as fr:
                temp_buffer = pickle.load(fr)
            self.buffer += temp_buffer
        
        
