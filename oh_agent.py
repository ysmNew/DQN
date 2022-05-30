import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from buffer import *

#Hyperparameters
learning_rate = 0.005
gamma         = 0.9
buffer_limit  = 50000
batch_size    = 128
#losses = []

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
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

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()

        self.conv1 = nn.Conv2d(7, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1920, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        #print('origin', x.shape)
        x = F.relu(self.conv1(x))
        #print('conv1', x.shape)
        x = F.relu(self.conv2(x))
        #print('conv1', x.shape)
        x = torch.flatten(x, 1)
        #print('flatten', x.shape)
        x = F.relu(self.fc1(x))
        #print('dence1', x.shape)
        x = self.fc2(x)
        #print('dence2', x.shape)
        return x
      
    def sample_action(self, obs, epsilon, action, goal_ob_reward):
        out = self.forward(obs)
        coin = random.random()
        if goal_ob_reward:
            if action == 0: return 1
            if action == 1: return 0
            if action == 2: return 3
            if action == 3: return 2
        elif coin < epsilon:
            act = random.randint(0,3)
            #print(out.detach().numpy()[0], act, 'Random!')
            return act
        else : 
            #print(out.detach().numpy()[0], out.argmax().item())
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, cr, s_prime, done_mask, gr = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        #losses.append(loss)
        optimizer.step()

