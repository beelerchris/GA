import gym
import gym.spaces
import numpy as np
from helper import *
from policy import Policy

n_gen = 10 # Number of generations
n_pop = 100 # Starting population
n_mutate = 5 # Number of mutations per generation
n_breed = 5 # Number of crossovers per generation
n_sacrifice = 15 # Number of removals per generation
hidden_units = np.array([128, 128]) # Number of kernels per layer, len(hidden_units) = number of layers
cross_p = 0.5 # Probability of policy1 weight being used during crossover
mut_p = 0.05 # Probability of weight mutating
wins = 10 # Number of wins to be considered the best

env = gym.make('CartPole-v0')
s0 = env.reset()
s0 = np.reshape(s0, (s0.shape[0], 1))
num_actions = int(env.action_space.n)

name = 0
if n_sacrifice > n_mutate + n_breed:
    n_sacrifice = n_mutate + n_breed
    print ('Sacrifice > growth per generation. n_sacrifice lowered to ' + str(n_sacrifice))

population = []
for i in range(n_pop):
    policy = Policy(s0, hidden_units, num_actions)
    policy.gen_random()
    population.append(policy)

scores = np.zeros(n_pop)

for gen in range(n_gen):
    scores = np.zeros(n_pop)
    for i in range(n_pop):
        scores[i] = evaluate_policy(population[i], env)

    print('Generation %d: Max Score = %0.2f, Population Size = %i' %(gen+1, scores.max(), n_pop))

    l1, l2 = zip(*sorted(zip(scores, population)))
    scores = np.array(l1[n_sacrifice:])
    population = list(l2[n_sacrifice:])
    population[-1].win += 1
    n_pop -= n_sacrifice
    younglings = []
    mutants = []
    for i in range(n_breed):
        policy1, policy2 = selection(population, scores)
        new_policy = crossover(policy1, policy2, cross_p)
        younglings.append(new_policy)

    n_pop += n_breed

    for i in range(n_mutate):
        policy1, policy2 = selection(population, scores)
        new_policy = mutation(policy1, mut_p)
        mutants.append(new_policy)

    n_pop += n_mutate

    population += younglings
    population += mutants

scores = np.zeros(n_pop)
for i in range(n_pop):
    scores[i] = evaluate_policy(population[i], env)

print('Best policy score = %0.2f.' %(np.max(scores)))

l1, l2 = zip(*sorted(zip(scores, population)))
champion = l2[-1]
champion.win += 1
np.savez('champion.npz', w=champion.W, b=champion.B)

print('Champion has won ' + str(champion.win) + ' game(s)!')
