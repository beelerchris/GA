import numpy as np
import gym
import gym.spaces
import cleangym
from grid_policy import Policy
from grid_helper import *

game = 'Carnot-v0'
gen = 100
data = np.load('./champions/' + game + '_' + str(100) + '.npz')

env = gym.make(game)
num_actions = int(env.action_space.n)

champion = Policy(num_actions)
#champion.policy = data['w']
champion.gen_perfect()

score = 0.0
for i in range(100):
    score += evaluate_policy(champion, env)

#score = score / 100.0
#print('Champion Average Score = ' + str(score))

vis_policy(champion, env)
