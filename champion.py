import numpy as np
import gym
import gym.spaces
from policy import Policy
from helper import *

game = 'CartPole-v0'
data = np.load('./champions/' + game + '.npz')

hidden_units = np.array([128, 128])
env = gym.make(game)
s0 = env.reset()
s0 = np.reshape(s0, (s0.shape[0], 1))
num_actions = int(env.action_space.n)

champion = Policy(s0, hidden_units, num_actions)
champion.W = data['w']
champion.B = data['b']

score = 0.0
for i in range(100):
    score += evaluate_policy(champion, env)

score = score / 100.0
print('Champion Average Score = ' + str(score))

for i in range(10):
    vis_policy(champion, env)
