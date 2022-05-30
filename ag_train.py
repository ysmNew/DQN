import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from agent import *
from Sim import *
#from buffer import *

def main():
    sim = Simulator()
    running_loss = 0.0
    q = Qnet()
    q.to(device)
    #print(next(q.parameters()).device)
    #input()
    q_target = Qnet()
    q_target.to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    #if os.listdir(bf_PATH):
        #memory = memory.load()

    learning_rate = 0.01
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for ep in range(len(sim.files)*10):
        #PATH = "state_dict_model_13_"+str(ep)+".pt"
        epi = ep%39999
        s = sim.reset(ep)
        epsilon = max(0.01, 0.99 - (ep/350000))
        if ep % 200000 == 0:
            learning_rate /= 10 # 0.001 0.0001 0.00001
        done = False

# 첫번째 액션을 0 으로 고정
        s, a, r, cr, s_prime, done, gr = sim.step(0)
        memory.put((s, a, r, cr, s_prime, done))
        s = s_prime

        while not done:
            for i in range(10):
                action = q.sample_action(s.float(), epsilon, a, gr)
                s, a, r, cr, s_prime, done, gr = sim.step(action)
                done_mask = 0.0 if done else 1.0
                memory.put((s, a, r, cr, s_prime, done_mask))
                s = s_prime

                if done:
                    print(sim.actions)
                    print('episode {}, epsilon {}, learning rate {}'.format(ep,epsilon,learning_rate))
                    print('lenth:', len(sim.actions), 'cr : ', cr)
                    print('===========================================================================')
                    break

        if memory.size()>40000:
            loss = train(q, q_target, memory, optimizer)
            #input()
            running_loss += loss
            if ep % 1000 == 0:
                q_target.load_state_dict(q.state_dict())
            if ep % 500 == 0:
                writer.add_scalar('training loss', running_loss/500, ep)
                running_loss = 0.0
                
        #if ep % 39999 == 0:
            #torch.save(q.state_dict(), PATH)
    
    PATH = "state_dict_model_13.pt"
    torch.save(q.state_dict(), PATH)                 
                    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('tensor in', device)
    input()
    writer = SummaryWriter('train_13')
    
    main()
    #plt.plot(losses)
