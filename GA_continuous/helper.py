import numpy as np
from policy import Policy
import gym
import gym.spaces

def selection(population, scores):
    p = scores / scores.sum()
    i = np.random.choice(range(len(population)), p = p)
    j = i
    while i == j:
        j = np.random.choice(range(len(population)), p = p)

    r = np.random.uniform()
    if r < 0.5:
        return population[i], population[j]
    else:
        return population[j], population[i]

def crossover(cross_pop, p = 0.5):
    policy1 = cross_pop[0]
    policy2 = cross_pop[1]
    new_policy = Policy(policy1.shape, policy1.hidden_units, policy1.num_actions, policy1.a_bound, policy1.game)
    for i in range(int(len(policy1.W) * p)):
        w = np.zeros((policy1.W[i].shape[0], policy1.W[i].shape[1]))
        b = np.zeros(policy1.B[i].shape[0])
        for j in range(len(policy1.W[i])):
            for k in range(len(policy1.W[i][j])):
                w[j][k] = policy1.W[i][j][k]

        for j in range(len(policy1.B[i])):
            b[j] = policy1.B[i][j]

        new_policy.W.append(w)
        new_policy.B.append(b)

    for i in range(int(len(policy2.W) * (1 - p)), len(policy2.W)):
        w = np.zeros((policy1.W[i].shape[0], policy1.W[i].shape[1]))
        b = np.zeros(policy1.B[i].shape[0])
        for j in range(len(policy2.W[i])):
            for k in range(len(policy2.W[i][j])):
                w[j][k] = policy2.W[i][j][k]

        for j in range(len(policy2.B[i])):
            b[j] = policy2.B[i][j]

        new_policy.W.append(w)
        new_policy.B.append(b)

    return new_policy

def mutation(mutate_pop, p = 0.5):
    policy = mutate_pop[0]
    new_policy = Policy(policy.shape, policy.hidden_units, policy.num_actions, policy.a_bound, policy.game)
    for i in range(len(policy.W)):
        w = np.zeros((policy.W[i].shape[0], policy.W[i].shape[1]))
        b = np.zeros(policy.B[i].shape[0])
        for j in range(len(policy.W[i])):
            for k in range(len(policy.W[i][j])):
                w[j][k] = policy.W[i][j][k] + p * np.random.normal()

        for j in range(len(policy.B[i])):
            b[j] = policy.B[i][j] + p * np.random.normal()

        new_policy.W.append(w)
        new_policy.B.append(b)

    return new_policy

def evaluate_policy(policy):
    game = policy.game
    env = gym.make(game)
    reward = 0
    for i in range(5):
        s = env.reset()
        d = False
        while not d:
            a = policy.evaluate(s)
            s, r, d, _ = env.step(a)
            reward += r

    return reward / 5.0

def evaluate_policy_single(policy):
    game = policy.game
    env = gym.make(game)
    reward = 0
    s = env.reset()
    d = False
    while not d:
        a = policy.evaluate(s)
        s, r, d, _ = env.step(a)
        reward += r

    return reward

def vis_policy(policy):
    game = policy.game
    env = gym.make(game)
    s = env.reset()
    env.render()
    d = False
    while not d:
        a = policy.evaluate(s)
        s, r, d, _ = env.step(a)
        env.render()
    env.close()
