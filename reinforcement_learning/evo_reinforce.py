import argparse
import gym
import numpy as np
from itertools import count
import itertools 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from torch import Tensor as TT

# TODO:
"""
Features:
    different recombinators
    recombine EVERYTHING
    compare 

"""


# TODO: with REINFORCE, we're only doing one update per epoch, so we need to recombine at a slower scale (or do more updates / epoch)!

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
#
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--recombine_every_n', type=int, default=50)
parser.add_argument('--n_episodes', type=int, default=50)
parser.add_argument('--recombinator', type=str, default='dropout')
parser.add_argument('--train_genes', type=int, default=1)
parser.add_argument('--train_gene_layers', type=int, default=1)
#
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()
args_dict = args.__dict__
locals().update(args_dict)

#
n_genes = 31
n_agents = 10
n_survivors = n_agents / 2
#recombine_every_n  = 5

env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 64)
        self.affine2 = nn.Linear(64, 2)

        # TODO: should this be trainable??
        self.genes = nn.Parameter(1. * torch.randn(1,n_genes))
        self.genes.name = 'genes'
        # these transform genes into CBN params
        self.gene_l1 = nn.Linear(n_genes, 64)
        self.gene_l2 = nn.Linear(64, 2*64)
        #self.phenotype = self.gene_l2(F.relu(self.gene_l1(self.genes)))

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        #x, genes = x[:, :4], x[:, 4:] 
        x = self.affine1(x)
        self.cbn = self.gene_l2(F.relu(self.gene_l1(self.genes)))
        x = x * (self.cbn[:,::2] + 1.) + self.cbn[:,1::2]
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores)




def recombine(agents):
    """
    For now, we'll recombine genes, and just average all the other params
    Then we can compare that with averaging genes, as well
    """
    n_kids = n_agents - n_survivors
    # TODO: sum across episodes (???)
    all_returns = [agent.returns for agent in agents]
    #print all_returns
    srts = np.argsort(all_returns)
    # agents with the highest fitness(=returns) survive
    fit_agents = [agents[srts[n]] for n in range(n_survivors)]
    unfit_agents = [agents[srts[n]] for n in range(n_survivors, n_agents)]
    # which of the parents to inheret from for each gene
    if recombinator == 'dropout':
        which_parents = np.random.binomial(n=1, p=.5, size=(n_kids, n_genes))
    elif recombinator == 'chromosome':
        which_parents = np.hstack((np.ones(n_genes/2), np.zeros(n_genes - n_genes/2)))
    # all possible couples
    possible_parents = list(itertools.combinations(range(n_survivors), 2))
    # convert them to sets because numpy is dumb
    possible_parents = [set(p) for p in possible_parents]
    # replace=True ==> couples can have multiple kids
    parents = np.random.choice(possible_parents, n_kids, replace=True)
    #print parents
    # ...and convert them BACK to lists :P
    parents = [list(p) for p in parents]
    kids_genes = [which_parent * fit_agents[parent[0]].genes.data.numpy() + (1 - which_parent) * fit_agents[parent[1]].genes.data.numpy() for which_parent, parent in zip(which_parents, parents) ] 
    #print kids_genes[0]
    # TODO: replace the bad ones genes with the kids genes
    for kid_genes, agent in zip(kids_genes, unfit_agents):
        agent.genes.data = TT(kid_genes)


# TODO: test me!
def sync(agents):
    """ sync all the unnamed params (i.e. all of them except the genes) """
    # [agents [n_params] ]
    params_to_average = [[p for p in agent.parameters() if not hasattr(p, 'name')] for agent in agents]
    # [n_params]
    average_params = [np.mean([pps[n].data.numpy() for pps in params_to_average], axis=0) for n in range(len(params_to_average[0]))]
    for n_param, avg_p in enumerate(average_params):
        for params in params_to_average:
            # set the params for this agent to be the average
            params[n_param].data = torch.Tensor(avg_p)


agents = [Policy() for _ in range(n_agents)]
optimizers = [optim.SGD(agent.parameters(), lr=5e-3, momentum=0) for agent in agents]
# TODO: adam
#optimizers = [optim.Adam(agent.parameters(), lr=1e-2) for agent in agents]


if 0:# TESTS
    for agent in agents:
        agent.rewards = [np.random.randn()]
    recombine(agents)
    print "recombined"
    sync(agents)
    assert False

if not args.resume:
    pass
    #agent = Policy()
    #optimizer = optim.Adam(agent.parameters(), lr=1e-2)


def select_action(agent, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = agent(Variable(state))
    action = probs.multinomial()
    agent.saved_actions.append(action)
    return action.data


def finish_episode(agent, optimizer, train_genes=True, train_gene_layers=True):
    R = 0
    rewards = []
    for r in agent.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(agent.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(agent.saved_actions, [None for _ in agent.saved_actions])
    if not train_genes:
        agent.genes.zero_grad()
    if not train_gene_layers:
        agent.gene_l1.zero_grad()
        agent.gene_l2.zero_grad()
    optimizer.step()
    agent.returns = len(agent.rewards)
    del agent.rewards[:]
    del agent.saved_actions[:]

if args.resume:
    del agent.rewards[:]
    del agent.saved_actions[:]


running_reward = 10
all_returns = np.zeros((n_episodes, n_agents))

for i_episode in range(n_episodes):#count(1):

    print i_episode
    if i_episode > 0 and i_episode % recombine_every_n == 0:
        print "\t\t\trecombining!"
        recombine(agents)

    for nn, (agent, optimizer) in enumerate(zip(agents, optimizers)):
        state = env.reset()

        done = 0
        t = 0
        #for t in range(10000): # Don't infinite loop while learning
        while not done:
            action = select_action(agent, state)
            state, reward, done, _ = env.step(action[0,0])
            if args.render:
                env.render()
            agent.rewards.append(reward)
            t += 1
            #if done:
            #    break

        # TODO: monitoring
        all_returns[i_episode, nn] = t
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode(agent, optimizer, train_genes, train_gene_layers)
        #if i_episode % args.log_interval == 0:
        if i_episode % recombine_every_n == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        #if running_reward > 200:
            #print("Solved! Running reward is now {} and "
                  #"the last episode runs to {} time steps!".format(running_reward, t))
            #break
    sync(agents)

np.save(save_path + 'all_returns.npy', all_returns)



