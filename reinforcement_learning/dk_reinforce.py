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
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()


gamma = 1.
num_seeds = 100
query_fns = ['p=0', 'p=.1', 'p=.5', 'p=1']
num_episodes = 1000
num_steps = 500

# TODO: query cost, timing

# -------------------------
print "LOGGING"
rewards_true = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
rewards_predicted = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
rewards_observed = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
queries = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
episode_lengths = np.zeros((len(query_fns), num_seeds, num_episodes))



# -------------------------
print "DEFINE MODEL CLASSES"
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
        self.memory = [] # each element is an ordered pair: (x, r)
        self.opt = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.affine2(x)

    # TODO: we could do this like the the Policy and always just train on the last batch of experience
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
        self.optimizer = optim.Adam(policy.parameters(), lr=1e-2)

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
            R = r + args.gamma * R
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

for seed in range(num_seeds):

    print "making a new random environment"
    #env = gym.make('CartPole-v1')
    from gym.dk_cartpole import CartPoleEnv
    if not args.resume:
        env = CartPoleEnv()
        # FIXME: seeding
        env.seed(seed)
        torch.manual_seed(seed)

    for nn, query_fn in enumerate(query_fns):

        if args.resume:
            print "RESUME"
            del policy.rewards[:]
            del policy.saved_actions[:]
        else:
            print "DECLARING MODELS"
            policy = Policy()
            reward_predictor = RewardPredictor()

        def finish_episode():
            reward_predictor.step()
            policy.step()
            del policy.rewards[:]
            del policy.saved_actions[:]


        running_reward = 0
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
                query = np.random.rand() < .5 # TODO
                query = 1
                query = 0

                if query:
                    policy.rewards.append(reward)
                    reward_predictor.memory.append([state[0], reward])
                else: # use the predicted reward instead
                    policy.rewards.append(reward_predictor.fn(np.array(state[0]).reshape((1,1))).data.numpy()[0][0])

                if done:
                    episode_lengths[nn, seed, episode] = step
                    break

            # TODO: track predicted and actual rewards separately!
            running_reward = running_reward * 0.99 + .01 * (sum(policy.rewards))
            finish_episode()
            if episode % args.log_interval == 0:
                print('Episode {}\tLast length: {:5d}\trunning average reward: {:.2f}'.format(episode, step, running_reward))
            if 0:#episode % 25 == 0:
                from pylab import *
                plot(env.discretized_reward_function, label='ground truth')
                learned_reward_function = reward_predictor(Variable(torch.from_numpy(np.linspace(-2.4, 2.4, 600).reshape((-1,1))).float())).data.numpy()
                plot(learned_reward_function, label='predicted')

        show()

