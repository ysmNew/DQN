import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from buffer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
gamma         = 0.99
buffer_limit  = 50000
batch_size    = 256

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, cr_lst, s_prime_lst, done_mask_lst = [], [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, cr, s_prime, done_mask = transition
            s = torch.squeeze(s,0)
            s_prime = torch.squeeze(s,0)
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            cr_lst.append([cr])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.stack(s_lst, dim=0), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(cr_lst), \
               torch.stack(s_prime_lst, dim=0), torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()

        self.conv1 = nn.Conv2d(8, 32, 3, 1) 
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1920, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = x.to(device)            # N, 8, 10, 9
        #print('origin', x.shape)
        x = F.relu(self.conv1(x))   # N, 32, 8, 7
        #print('conv1', x.shape)
        x = F.relu(self.conv2(x))   # N, 64, 6, 5
        #print('conv1', x.shape)
        x = torch.flatten(x, 1)     # N, 64, 30
        #print('flatten', x.shape)
        x = F.relu(self.fc1(x))     # N, 1920 -> N, 128
        #print('dence1', x.shape)
        x = self.fc2(x)             # N, 128 -> N, 4
        #print('dence2', x.shape)
        return x
      
    def sample_action(self, obs, epsilon, action, goal_ob_reward):
        out = self.forward(obs)
        coin = random.random()
# 아이템을 먹고나면 반드시 빠져나오도록 세팅
        if coin < epsilon:
            act = random.randint(0,3)
            #print(out.detach().numpy()[0], act, 'Random!')
            return act
        else : 
            #print(out.detach().numpy()[0], out.argmax().item())
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, cr, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a.to(device))
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r.to(device) + gamma * max_q_prime * done_mask.to(device)
        #print(target.device)
        #input()
        loss = F.smooth_l1_loss(q_a, target.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

