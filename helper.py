import numpy as np
from policy import Policy

def selection(population, scores):
    p = scores / scores.sum()
    i = np.random.choice(range(len(population)), p = p)
    j = i
    while i == j:
        j = np.random.choice(range(len(population)), p = p)

    return population[i], population[j]

def crossover(policy1, policy2, p = 0.5):
    new_policy = Policy(policy1.state, policy1.hidden_units, policy1.num_actions)
    for i in range(len(policy1.W)):
        w = np.zeros((policy1.W[i].shape[0], policy1.W[i].shape[1]))
        b = np.zeros(policy1.B[i].shape[0])
        for j in range(len(policy1.W[i])):
            for k in range(len(policy1.W[i][j])):
                r = np.random.uniform()
                if r < p:
                    w[j][k] = policy1.W[i][j][k]
                else:
                    w[j][k] = policy2.W[i][j][k]
        for j in range(len(policy1.B[i])):
            r = np.random.uniform()
            if r < 0.5:
                b[j] = policy1.B[i][j]
            else:
                b[j] = policy2.B[i][j]
        new_policy.W.append(w)
        new_policy.B.append(b)

    return new_policy

def mutation(policy, p = 0.05):
    new_policy = Policy(policy.state, policy.hidden_units, policy.num_actions)
    for i in range(len(policy.W)):
        w = np.zeros((policy.W[i].shape[0], policy.W[i].shape[1]))
        b = np.zeros(policy.B[i].shape[0])
        for j in range(len(policy.W[i])):
            for k in range(len(policy.W[i][j])):
                r = np.random.uniform()
                if r < p:
                    w[j][k] = np.random.normal()
                else:
                    w[j][k] = policy.W[i][j][k]
        for j in range(len(policy.B[i])):
            r = np.random.uniform()
            if r < p:
                b[j] = np.random.normal()
            else:
                b[j] = policy.B[i][j]
        new_policy.W.append(w)
        new_policy.B.append(b)

    return new_policy

def evaluate_policy(policy, env):
    reward = 0
    s = env.reset()
    s = np.reshape(s, (s.shape[0], 1))
    d = False
    while not d:
        a = policy.evaluate(s)
        s, r, d, _ = env.step(a)
        s = np.reshape(s, (s.shape[0], 1))
        reward += r

    return reward

def vis_policy(policy, env):
    s = env.reset()
    env.render()
    s = np.reshape(s, (s.shape[0], 1))
    d = False
    while not d:
        a = policy.evaluate(s)
        s, r, d, _ = env.step(a)
        env.render()
        s = np.reshape(s, (s.shape[0], 1))
