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
#   

# TODO: recall that QUERY_COST and EXPERIMENT_LENGTH are important to consider!!!

# TODO: also monitor reward function error (x2)


"""
QUERY PREDICTION ALGORITHMS:
    * improvement in predictive ability
    * adversarial (Sherjil)
        every time a new state is encountered, we add a "new state, old state" pair to the training set; the goal is to classify them both correctly

    * model-based!?

    * meta-learning ;)
    * some AL thing?
"""

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
save_dir = 'results1/analyzed/'
import os
os.mkdir(save_dir)
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






