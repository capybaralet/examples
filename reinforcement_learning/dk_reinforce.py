import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import time

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()


# each experiment takes ~10-20s, so I can do ~200/hr
gamma = .99
num_seeds = 30
query_fns = []
query_fns += ['p=0', 'p=.1', 'p=.5', 'p=1']
query_fns += ['N=100', 'N=1000', 'N=10000']
num_episodes = 300
num_steps = 300


# -------------------------
print "LOGGING"
save_dir = 'results1/'
reward_err_experience = np.zeros((len(query_fns), num_seeds, num_episodes))
reward_err_uniform = np.zeros((len(query_fns), num_seeds, num_episodes))
rewards_true = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
rewards_predicted = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
rewards_observed = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
queries = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
episode_lengths = np.zeros((len(query_fns), num_seeds, num_episodes))



# -------------------------
print "DEFINE MODEL CLASSES"
# TODO: move these to modules


# TODO: sherjil
class StateDiscriminator(nn.Module):
    """
    Here, we try and
    """
    def __init__(self):
        super(StateDiscriminator, self).__init__()
        self.affine1 = nn.Linear(4, 64)
        self.affine2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return F.sigmoid(self.affine2(x))

    def train(self, pos, neg, mbsz=128):
        opt = optim.Adam(self.parameters(), lr=1e-2)
        for i in xrange(1000):
            posidx = np.random.choice(len(pos), (mbsz/2,))
            negidx = np.random.choice(len(neg), (mbsz/2,))
            posmbx = Variable(torch.from_numpy(pos[posidx]).float())
            negmbx = Variable(torch.from_numpy(neg[negidx]).float())
            opt.zero_grad()
            posmby = self(posmbx)
            negmby = self(negmbx)
            loss = (negmby - posmby).mean()
            loss.backward()
            opt.step()
        print "rewardfn loss", loss.data[0]

    def fn(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        return self(state)
        #return F.logsigmoid(preact)


class RewardTeacher(nn.Module):
    """
    The idea here is to reward this model based on it's demonstrated
    """
    pass

class RewardPredictor(nn.Module):
    """
    There's some details to be worked out here.... we want to train on all data points an approximately equal number of times...
    For now, we'll just keep all observations in memory, and train on a random subset every time.
    """
    def __init__(self, batch_size=32):
        super(RewardPredictor, self).__init__()
        self.batch_size = batch_size
        self.affine1 = nn.Linear(1, 64)
        self.affine2 = nn.Linear(64, 1)
        # all rewards ever
        self.memory = [] # each element is an ordered pair: (x, r)
        # just the last episode
        self.rewards = [] # each element is an ordered pair: (x, r)
        self.opt = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.affine2(x)

    # TODO: clean-up
    def step(self):
        if len(self.rewards) > 0:
            examples = self.rewards
            inputs = np.array([[ex[0] for ex in examples]]).T
            inputs = Variable(torch.from_numpy(inputs).float())
            targets = np.array([[ex[1] for ex in examples]]).T
            targets = Variable(torch.from_numpy(targets).float())

            self.opt.zero_grad()
            loss = ((self.forward(inputs) - targets)**2).mean()
            loss.backward()
            self.opt.step()
            self.memory += self.rewards
            del self.rewards[:]

    def memory_step(self):
        if len(self.memory) >= self.batch_size:
            inds = np.random.choice(len(self.memory), self.batch_size)
            examples = [self.memory[ind] for ind in inds]
            inputs = np.array([[ex[0] for ex in examples]]).T
            inputs = Variable(torch.from_numpy(inputs).float())
            targets = np.array([[ex[1] for ex in examples]]).T
            targets = Variable(torch.from_numpy(targets).float())
            self.opt.zero_grad()
            loss = ((self.forward(inputs) - targets)**2).mean()
            loss.backward()
            self.opt.step()

    def loss(self, dataset):
        if dataset == 'experience':
            if len(self.memory) > 600:
                inds = np.random.choice(len(self.memory),600)
            else:
                inds = range(len(self.memory))
            examples = [self.memory[ind] for ind in inds]
        else:
            examples = dataset

        inputs = np.array([[ex[0] for ex in examples]]).T
        inputs = Variable(torch.from_numpy(inputs).float())
        targets = np.array([[ex[1] for ex in examples]]).T
        targets = Variable(torch.from_numpy(targets).float())
        loss = ((self.forward(inputs) - targets)**2).mean()
        return loss.data.numpy()

    def fn(self, state):
        state = torch.from_numpy(state).float()
        return self(Variable(state))


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 64)
        self.affine2 = nn.Linear(64, 2)
        self.saved_actions = []
        self.rewards = []
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)

    def fn(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self(Variable(state))
        action = probs.multinomial()
        self.saved_actions.append(action)
        return action.data

    def step(self):
        R = 0
        rewards = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        # TODO: what is up with the normalization by std??
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for action, r in zip(self.saved_actions, rewards):
            action.reinforce(r)
        self.optimizer.zero_grad()
        autograd.backward(self.saved_actions, [None for _ in self.saved_actions])
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]


# -------------------------
print "TRAINING"

total_nexp = num_seeds * len(query_fns)
nexp = 0
t0 = time.time()

for seed in range(num_seeds):

    print "making a new random environment"
    #env = gym.make('CartPole-v1')
    from gym.dk_cartpole import CartPoleEnv
    if not args.resume:
        env = CartPoleEnv()
        # FIXME: seeding
        env.seed(seed)
        torch.manual_seed(seed)

    for nq, query_fn in enumerate(query_fns):
        t1 = time.time()
        nexp += 1
        print "begin experiment #", nexp, "out of", total_nexp

        if args.resume:
            print "RESUME"
            del policy.rewards[:]
            del policy.saved_actions[:]
        else:
            #print "DECLARING MODELS"
            policy = Policy()
            reward_predictor = RewardPredictor()

        def finish_episode():
            reward_predictor.step()
            for i in range(5):
                reward_predictor.memory_step()
            policy.step()
            del policy.rewards[:]
            del policy.saved_actions[:]


        running_reward = 0
        num_queries = 0
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = policy.fn(state)
                state, reward, done, _ = env.step(action[0,0])

                if args.render:
                    if step == 0:
                        learned_reward_function = reward_predictor(Variable(torch.from_numpy(np.linspace(-2.4, 2.4, 600).reshape((-1,1))).float())).data.numpy()
                    else:
                        learned_reward_function = None
                    env._render(learned_reward_function=learned_reward_function)
                
                # --------------------
                # QUERY???
                if query_fn.startswith('p'):
                    query = np.random.rand() < float(query_fn.split('=')[1])
                elif query_fn.startswith('N'):
                    query = num_queries < int(query_fn.split('=')[1])

                predicted_reward = reward_predictor.fn(np.array(state[0]).reshape((1,1))).data.numpy()[0][0]

                if query:
                    num_queries += 1
                    policy.rewards.append(reward)
                    reward_predictor.rewards.append([state[0], reward])
                else: # use the predicted reward instead
                    policy.rewards.append(predicted_reward)

                # --------------------
                # LOGGING
                rewards_true[nq, seed, episode, step] = reward
                rewards_predicted[nq, seed, episode, step] = predicted_reward
                rewards_observed[nq, seed, episode, step] = reward if query else predicted_reward
                queries[nq, seed, episode, step] = query

                if done:
                    break


            # TODO: change this output
            running_reward = running_reward * 0.99 + .01 * (sum(policy.rewards))

            # TRAINING STEPS
            finish_episode()

            # MORE LOGGING
            episode_lengths[nq, seed, episode] = step
            if not len(reward_predictor.memory) == 0:
                reward_err_experience[nq, seed, episode] = reward_predictor.loss('experience')
            reward_err_uniform[nq, seed, episode] = reward_predictor.loss(zip(np.linspace(-2.4, 2.4, 600), env.discretized_reward_function))

            if episode % 10 == 0:
                if args.verbose:
                    print('Episode {}\tLast length: {:5d}\trunning average reward: {:.2f}'.format(episode, step, running_reward))
            if 0:#episode % 25 == 0:
                from pylab import *
                plot(env.discretized_reward_function, label='ground truth')
                learned_reward_function = reward_predictor(Variable(torch.from_numpy(np.linspace(-2.4, 2.4, 600).reshape((-1,1))).float())).data.numpy()
                plot(learned_reward_function, label='predicted')

        print "time this run=", time.time() - t1,   "         total time=", time.time() - t0

    print "SAVING..."
    np.save(save_dir + 'reward_err_experience',reward_err_experience)
    np.save(save_dir + 'reward_err_uniform',reward_err_uniform)
    np.save(save_dir + 'rewards_true',rewards_true)
    np.save(save_dir + 'rewards_predicted',rewards_predicted)
    np.save(save_dir + 'rewards_observed',rewards_observed)
    np.save(save_dir + 'queries',queries )
    np.save(save_dir + 'episode_lengths',episode_lengths)

