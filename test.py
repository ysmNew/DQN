import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent import *
from Sim import *

def test_main():
    sim = Simulator(test=True)
    q = Qnet()
    q.to(device)
    PATH = "state_dict_model_13.pt"
    q.load_state_dict(torch.load(PATH))
    timestep = 0
    finish_num = 0

    actions_file_name='test_13'+'.txt'
    f = open(actions_file_name, 'w')

    for epi in range(1226):
        s = sim.reset(epi)
        #print(s[0][4])
        s, a, r, cr, s_prime, done, gr = sim.step(0)
        s = s_prime
        #print(s_prime[0][4])
        epsilon = 0
        done = False
        while not done:
            action = q.sample_action(s.float(), epsilon, a, gr)
            s, a, r, cr, s_prime, done, gr = sim.step(action)
            #print(s_prime[0][4])
            s = s_prime
            timestep += 1
            if gr == 'finish':
                finish_num += 1
            if done:
                print(sim.origin_target)
                print(sim.actions)
                print('episode {}, epsilon {}'.format(epi,epsilon))
                print('lenth:', len(sim.actions), 'cr : ', cr)
                break
        print('Episode :', epi, 'Timestep :', timestep, 'Reward :', cr, 'Finish Rate :', finish_num/1226)
        print('==============================================================================')

        if len(sim.actions)>3:
            f.write(str(epi)+'/'+str(sim.origin_target)+'/'+str(cr)+'/'+str(len(sim.actions))+'\n')
            f.write(str(sim.actions)+'\n')


    f.close()

if __name__ == '__main__':
    test_main()

