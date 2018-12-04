import gym
import gym.spaces
import numpy as np
from helper import *
from policy import Policy
from multiprocessing import Pool
import os
from reactive_control import rc_gym

if not os.path.exists('./champions'):
    os.makedirs('./champions')

n_gen = 1000 # Number of generations
n_pop = 100 # Starting population
n_mutate = 75 # Number of mutations per generation
n_breed = 0 # Number of crossovers per generation
n_sacrifice = 75 # Number of removals per generation
hidden_units = np.array([256]) # Number of kernels per layer, len(hidden_units) = number of layers
game = 'MountainCarContinuous-v0' # Game to play
cpus = 40 # Number of processes to run
load = False # Load previous champion
load_gen = 0 # Generation to load
wins = 200 # Wins required for champion to be considered winner

pool = Pool(processes = cpus)

env = gym.make(game)
s0 = env.reset()
shape = s0.shape[0]
num_actions = env.action_space.shape[0]
a_bound = [env.action_space.low, env.action_space.high]

name = 0
if n_sacrifice > n_mutate + n_breed:
    n_sacrifice = n_mutate + n_breed
    print ('Sacrifice > growth per generation. n_sacrifice lowered to ' + str(n_sacrifice))
if n_pop <= 2:
    n_sacrifice = 0
    print ('Not enough population to sacrifice. n_sacrifice lowered to ' + str(n_sacrifice))
elif n_sacrifice >= n_pop - 1:
    n_sacrifice = n_pop - 2
    print ('Sacrifice too large. n_sacrifice lowered to ' + str(n_sacrifice))

population = []
if load == True:
    gen = load_gen
    data = np.load('./champions/' + game + '_' + str(gen) + '.npz')
    hidden_units = data['h']
    policy = Policy(shape, hidden_units, num_actions, a_bound, game)
    population.append(policy)
    population[0].W = data['w']
    population[0].B = data['b']
    print ('Loading previous champion...')
else:
    gen = 0
    policy = Policy(shape, hidden_units, num_actions, a_bound, game)
    policy.gen_random()
    population.append(policy)

for i in range(n_pop - 1):
    policy = Policy(shape, hidden_units, num_actions, a_bound, game)
    policy.gen_random()
    population.append(policy)

winning = False
max_s0 = -10000.0

while not winning:
    for generation in range(n_gen):
        gen += 1
        scores = pool.map(evaluate_policy, population)

        scores = np.array(scores)

        l1, l2 = zip(*sorted(zip(scores, population), key = lambda x: x[0]))
        scores = np.array(l1[n_sacrifice:])
        population = list(l2[n_sacrifice:])
        population[-1].win += 1
        print('Generation %d: Average Score = %0.4f, Max Score = %0.4f, Policy Wins = %i, Population Size = %i' %(gen, np.mean(scores), scores[-1], population[-1].win, n_pop))
        n_pop -= n_sacrifice

        if max_s0 < scores[-1]:
            champion = population[-1]
            np.savez('./champions/' + game + '_' + str(gen) + '.npz', w=champion.W, b=champion.B, h=champion.hidden_units)
            print('Champion has won ' + str(champion.win + 1) + ' game(s)!')
            max_s0 = scores[-1]

        choice = np.ones(len(scores))
        cross_pop = []
        mutate_pop = []
        for i in range(n_breed):
            policy1, policy2 = selection(population, choice)
            cross_pop.append([policy1, policy2])

        for i in range(n_mutate):
            policy1, policy2 = selection(population, choice)
            mutate_pop.append([policy1])

        younglings = pool.map(crossover, cross_pop)
        mutants = pool.map(mutation, mutate_pop)

        n_pop += n_breed
        n_pop += n_mutate

        population += younglings
        population += mutants

    scores = pool.map(evaluate_policy, population)
    scores = np.array(scores)

    print('Best policy score = %0.2f.' %(np.max(scores)))

    l1, l2 = zip(*sorted(zip(scores, population), key = lambda x: x[0]))
    population = list(l2)
    champion = population[-1]
    champion.win += 1
    np.savez('./champions/' + game + '.npz', w=champion.W, b=champion.B, h=champion.hidden_units)
    print('Champion has won ' + str(champion.win) + ' game(s)!')

    if champion.win > wins:
        winning = True
