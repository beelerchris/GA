import gym
import gym.spaces
import numpy as np
from helper import *
from policy import Policy

n_pop = 10
n_steps = 10
n_elite = 5
hidden_units = np.array([128, 128])
cross_p = 0.5
mut_p = 0.05

env = gym.make('CartPole-v0')
s0 = env.reset()
s0 = np.reshape(s0, (s0.shape[0], 1))
num_actions = int(env.action_space.n)

population = []
for i in range(n_pop):
    policy = Policy(s0, hidden_units, num_actions)
    policy.gen_random()
    population.append(policy)

scores = np.zeros(n_pop)

for step in range(n_steps):
    for i in range(n_pop):
        scores[i] = evaluate_policy(population[i], env)

    print('Generation %d: Max Score = %0.2f' %(step+1, scores.max()))

    print scores
    ranks = n_pop - 1 - np.argsort(np.argsort(scores))
    print ranks
    new_population = list(population)
    for i in range(n_pop):
        if ranks[i] >= n_elite:
            policy1, policy2 = selection(population, scores)
            new_policy = Policy(s0, hidden_units, num_actions)
            crossover(policy1, policy2, new_policy, cross_p)
            new_population[i] = new_policy

        mutation(new_population[i], mut_p)

for i in range(n_pop):
    scores[i] = evaluate_policy(population[i], env)

print('Best policy score = %0.2f.' %(np.max(scores)))
