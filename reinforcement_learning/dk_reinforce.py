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


"""
So far, I've modified cart-pole's rewards, and now they show up in the rendering (cool!)
    It would be cool if I could also get the learned reward function to be displayed...
    I should ALSO make the reward function a bit NOISY!!!

I should probably also register the environment or do some other hack so that it can't balance forever

The next step should be to implement ARL stuff...

"""


# FOR LOOP:
#   seed (REMEMBER TO PUT THIS IN THE OUTER LOOP!!!)
#   query_fn
#   environment??



"""
QUERY PREDICTION ALGORITHMS:
    1. always
    2. never
    3. p = .5
    4. first N times
    ----------------
    5. some AL thing?
    6. improvement in predictive ability
    7. adversarial (Sherjil)
    8. meta-learning ;)

    model-based!?

"""


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
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()


#env = gym.make('CartPole-v1')
from gym.dk_cartpole import CartPoleEnv
if not args.resume:
    env = CartPoleEnv()
    # FIXME: seeding
    env.seed(args.seed)
    torch.manual_seed(args.seed)


class RewardPredictor(nn.Module):
    """
    There's some details to be worked out here.... we want to train on all data points an approximately equal number of times...
    For now, we'll just keep all observations in memory, and train on a random subset every time.
    """
    def __init__(self, batch_size=32):
        # TODO: does this come before or after super?
        self.__dict__.update(locals())

        super(RewardPredictor, self).__init__()

        self.affine1 = nn.Linear(1, 64)
        self.affine2 = nn.Linear(64, 1)
        self.memory = [] # each element is an ordered pair: (x, r)
        self.opt = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.affine2(x)

    def step(self):
        if len(self.memory) >= self.batch_size:
            inds = np.random.choice( len(self.memory), self.batch_size)
            examples = [self.memory[ind] for ind in inds]
            inputs = np.array([[ex[0] for ex in examples]]).T
            inputs = Variable(torch.from_numpy(inputs).float())
            targets = np.array([[ex[1] for ex in examples]]).T
            targets = Variable(torch.from_numpy(targets).float())
            self.opt.zero_grad()
            loss = ((self.forward(inputs) - targets)**2).mean()
            loss.backward()
            self.opt.step()
            #print "rewardfn loss", loss.data[0]
            

    def fn(self, state):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        state = torch.from_numpy(state).float()#.unsqueeze(0)
        return self(Variable(state))




# below here could be cleaned up...
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 64)
        self.affine2 = nn.Linear(64, 2)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


if not args.resume:
    policy = Policy()
    reward_predictor = RewardPredictor()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    action = probs.multinomial()
    policy.saved_actions.append(action)
    return action.data


# TODO: should also train reward predictor after each episode
def finish_episode():
    # REINFORCE
    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(policy.saved_actions, rewards):
        action.reinforce(r)

    optimizer.zero_grad()
    autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]
    for nstep in range(3):
        reward_predictor.step()




if args.resume:
    del policy.rewards[:]
    del policy.saved_actions[:]



# -------------------------
print "LOGGING"
rewards_true
rewards_predicted
rewards_observed
queries
episode_length




running_reward = 0
for i_episode in range(1000):#count(1):
    state = env.reset()
    for t in range(500): # episode_length
        action = select_action(state)
        state, reward, done, _ = env.step(action[0,0])
        if args.render:
            if t == 0:
                learned_reward_function = reward_predictor(Variable(torch.from_numpy(np.linspace(-2.4, 2.4, 600).reshape((-1,1))).float())).data.numpy()
            else:
                learned_reward_function = None
            env._render(learned_reward_function=learned_reward_function)
        
        # --------------------
        # QUERY???
        query = np.random.rand() < .5 # TODO
        query = 1
        query = 0

        if query:
            policy.rewards.append(reward)
            reward_predictor.memory.append([state[0], reward])
        else: # use the predicted reward instead
            policy.rewards.append(reward_predictor.fn(np.array(state[0]).reshape((1,1))).data.numpy()[0][0])

        if done:
            break

    # TODO: track predicted and actual rewards separately!
    running_reward = running_reward * 0.99 + .01 * (sum(policy.rewards))
    finish_episode()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\trunning average reward: {:.2f}'.format(i_episode, t, running_reward))
    if 0:#i_episode % 25 == 0:
        from pylab import *
        plot(env.discretized_reward_function, label='ground truth')
        learned_reward_function = reward_predictor(Variable(torch.from_numpy(np.linspace(-2.4, 2.4, 600).reshape((-1,1))).float())).data.numpy()
        plot(learned_reward_function, label='predicted')

show()

