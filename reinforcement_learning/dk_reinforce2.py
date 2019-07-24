import argparse
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import time

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# each experiment takes ~10-20s, so I can do ~200/hr
gamma = .99
num_seeds = 30
query_fns = []
#query_fns += ['bayes=.3', 'bayes=.1', 'bayes=.03']
query_fns += ['bayes=.1']
#query_fns += ['disc=.8', 'disc=.5', 'disc=.2']
query_fns += ['disc=.5']
num_episodes = 300
num_steps = 300
assert num_steps > 32 # TODO?

# -------------------------
print "LOGGING"
save_dir = 'results2/'
reward_err_experience = np.zeros((len(query_fns), num_seeds, num_episodes))
reward_err_uniform = np.zeros((len(query_fns), num_seeds, num_episodes))
rewards_true = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
rewards_predicted = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
rewards_observed = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
queries = np.zeros((len(query_fns), num_seeds, num_episodes, num_steps))
episode_lengths = np.zeros((len(query_fns), num_seeds, num_episodes))

# -------------------------
print "DEFINE MODEL CLASSES"
from modules import *

# -------------------------
print "TRAINING"

total_nexp = num_seeds * len(query_fns)
nexp = 0
t0 = time.time()

for seed in range(num_seeds):

    np.random.seed(seed)
    print "making a new random environment"
    from gym.dk_cartpole import CartPoleEnv
    env = gym.make('DKCartPole-v0')
    env.seed(seed)
    torch.manual_seed(seed)

    for nq, query_fn in enumerate(query_fns):
        t1 = time.time()
        nexp += 1
        print "begin experiment #", nexp, "out of", total_nexp

        thresh = float(query_fn.split('=')[1])

        if query_fn.startswith('bayes'):
            dropout=True
        else:
            dropout=False
            discriminator = StateDiscriminator()

        policy = Policy()
        reward_predictor = RewardPredictor(dropout=dropout)

        def finish_episode():
            if not dropout:
                discriminator.step() # this does 5 steps, internally
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

                if not dropout:
                    #print len(discriminator.unseen)
                    discriminator.unseen.append(state[0])

                if args.render:
                    if step == 0:
                        reward_predictor.training=0
                        learned_reward_function = reward_predictor(Variable(torch.from_numpy(np.linspace(-2.4, 2.4, 600).reshape((-1,1))).float())).data.numpy()
                        reward_predictor.training=1
                    else:
                        learned_reward_function = None
                    env._render(learned_reward_function=learned_reward_function)
                
                # --------------------
                # QUERY???
                if dropout:
                    noisy_predicted_rewards = [reward_predictor.fn(np.array(state[0]).reshape((1,1)))[0][0] for n in range(10)]
                    query = np.std(noisy_predicted_rewards) > thresh
                else:
                    prob_unseen = discriminator.fn(np.array(state[0]).reshape((1,1)))
                    query = prob_unseen > thresh

                reward_predictor.training=0
                predicted_reward = reward_predictor.fn(np.array(state[0]).reshape((1,1)))[0][0]
                reward_predictor.training=1

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
            reward_predictor.training=0
            if not len(reward_predictor.memory) == 0:
                reward_err_experience[nq, seed, episode] = reward_predictor.loss('experience')
            reward_err_uniform[nq, seed, episode] = reward_predictor.loss(zip(np.linspace(-2.4, 2.4, 600), env.discretized_reward_function))
            reward_predictor.training=1

            if args.verbose and episode % 10 == 0:
                print('Episode {}\tLast length: {:5d}\trunning average reward: {:.2f}'.format(episode, step, running_reward))

        print "time this run=", time.time() - t1,   "         total time=", time.time() - t0

    if save_dir is not None:
        print "SAVING..."
        np.save(save_dir + 'reward_err_experience',reward_err_experience)
        np.save(save_dir + 'reward_err_uniform',reward_err_uniform)
        np.save(save_dir + 'rewards_true',rewards_true)
        np.save(save_dir + 'rewards_predicted',rewards_predicted)
        np.save(save_dir + 'rewards_observed',rewards_observed)
        np.save(save_dir + 'queries',queries )
        np.save(save_dir + 'episode_lengths',episode_lengths)

