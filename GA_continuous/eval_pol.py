import numpy as np
import gym
import gym.spaces
from policy import Policy
from helper import *
import matplotlib.pyplot as plt
from multiprocessing import Pool

game = '2h2o-v0'
gen = 1
data = np.load('./champions/' + game + '/' + game + '_' + str(gen) + '.npz')
cpus = 4

env = gym.make(game)
s0 = env.reset()
shape = s0.shape[0]
num_actions = env.action_space.shape[0]
a_bound = [env.action_space.low, env.action_space.high]

hidden_units = data['h']

champion = Policy(shape, hidden_units, num_actions, a_bound, game)
champion.W = data['w']
champion.B = data['b']

pool = Pool(processes = cpus)
champions = []
for k in range(5):
    champions.append(champion)

scores = pool.map(evaluate_policy_single, champions)
scores = np.array(scores)
score = np.mean(score)

print('Champion Average Score = ' + str(score))

vis_policy(champion)
