import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class StateDiscriminator(nn.Module):
    """
    Very similar to reward predictor; the main differences are in the construction of the training data.
    I *still* think it's really weird that the label just flips in a very predictable way... 
    """
    def __init__(self, batch_size=32, dropout=False):
        super(StateDiscriminator, self).__init__()
        self.batch_size = batch_size
        self.dropout = dropout
        self.affine1 = nn.Linear(1, 64)
        self.affine2 = nn.Linear(64, 1)
        # all previous episodes
        self.seen = []
        # the most recent episode
        self.unseen = []
        self.opt = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        return F.sigmoid(self.affine2(x))

    def step(self):
        if len(self.seen) < self.batch_size and len(self.unseen) > self.batch_size: # we don't have enough seen data yet
            self.seen += self.unseen
            del self.unseen[:]
        elif len(self.seen) > self.batch_size and len(self.unseen) > self.batch_size:
            for i in range(5):
                # construct balanced mini-batch
                shuffles = np.random.permutation(2 * self.batch_size)
                inds_unseen = np.random.choice(len(self.unseen), self.batch_size)
                inds_seen = np.random.choice(len(self.seen), self.batch_size)
                inputs = np.array([self.seen[ind] for ind in inds_seen] + [self.unseen[ind] for ind in inds_unseen])
                targets = np.hstack((np.zeros(self.batch_size), np.ones(self.batch_size)))
                inputs = inputs[shuffles].reshape((-1,1))
                targets = targets[shuffles].reshape((-1,1))
                #import ipdb; ipdb.set_trace()
                inputs = Variable(torch.from_numpy(inputs).float())
                targets = Variable(torch.from_numpy(targets).float())

                self.opt.zero_grad()
                # TODO: cross-entropy??
                loss = ((self.forward(inputs) - targets)**2).mean()
                loss.backward()
                self.opt.step()

            self.seen += self.unseen
            del self.unseen[:]

    # TODO
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
        targets = np.array([[ex[1] for ex in examples]]).T
        inputs = Variable(torch.from_numpy(inputs).float())
        targets = Variable(torch.from_numpy(targets).float())
        loss = ((self.forward(inputs) - targets)**2).mean()
        return loss.data.numpy()

    def fn(self, state):
        state = torch.from_numpy(state).float()
        return self(Variable(state)).data.numpy()

class RewardPredictor(nn.Module):
    """
    There's some details to be worked out here.... we want to train on all data points an approximately equal number of times...
    For now, we'll just keep all observations in memory, and train on a random subset every time.
    """
    def __init__(self, batch_size=32, dropout=False):
        super(RewardPredictor, self).__init__()
        self.batch_size = batch_size
        self.dropout = dropout
        self.affine1 = nn.Linear(1, 64)
        self.affine2 = nn.Linear(64, 1)
        # all rewards ever
        self.memory = [] # each element is an ordered pair: (x, r)
        # just the last episode
        self.rewards = [] # each element is an ordered pair: (x, r)
        self.opt = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        return self.affine2(x)

    # TODO: clean-up
    def step(self):
        if len(self.rewards) > 0:
            examples = self.rewards
            inputs = np.array([[ex[0] for ex in examples]]).T
            #print inputs.shape
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
        return self(Variable(state)).data.numpy()


class RewardTeacher(nn.Module):
    """
    The idea here is to reward this model based on it's demonstrated ability to improve the predictions of the reward_predictor
    """
    pass


class Policy(nn.Module):
    def __init__(self, gamma=.99):
        super(Policy, self).__init__()
        self.gamma = gamma
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
            R = r + self.gamma * R
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

