import numpy as np
import gym
import gym.spaces
from policy import Policy
from helper import *
import matplotlib.pyplot as plt

game = 'MountainCarContinuous-v0'
gen = 161
data = np.load('./champions/' + game + '_' + str(gen) + '.npz')

env = gym.make(game)
s0 = env.reset()
shape = s0.shape[0]
num_actions = env.action_space.shape[0]
a_bound = [env.action_space.low, env.action_space.high]

hidden_units = data['h']

champion = Policy(shape, hidden_units, num_actions, a_bound, game)
champion.W = data['w']
champion.B = data['b']

score = 0.0
for k in range(10):
    score += evaluate_policy(champion)

score = score / 10.0
print('Champion Average Score = ' + str(score))

vis_policy(champion)
