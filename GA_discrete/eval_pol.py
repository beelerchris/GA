import numpy as np
import gym
import gym.spaces
from policy import Policy
from helper import *
import matplotlib.pyplot as plt
from multiprocessing import Pool

game = 'Acrobot-v1'
gen = 88
data = np.load('./champions/' + game + '/' + game + '_' + str(gen) + '.npz')
cpus = 4

env = gym.make(game)
s0 = env.reset()
shape = s0.shape[0]
num_actions = env.action_space.n

hidden_units = data['h']

champion = Policy(shape, hidden_units, num_actions, game)
champion.W = data['w']
champion.B = data['b']

pool = Pool(processes = cpus)
champions = []
for k in range(5):
    champions.append(champion)

scores = pool.map(evaluate_policy, champion)
scores = np.array(scores)
score = np.mean(score)

print('Champion Average Score = ' + str(score))

vis_policy(champion)
