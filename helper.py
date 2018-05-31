import numpy as np

def selection(population, scores):
    p = scores / scores.sum()
    i = np.random.choice(range(len(population)), p = p)
    j = i
    while i == j:
        j = np.random.choice(range(len(population)), p = p)
    return population[i], population[j]

def crossover(policy1, policy2, new_policy, p = 0.5):
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

def mutation(policy, p = 0.05):
    for i in range(len(policy.W)):
        for j in range(len(policy.W[i])):
            for k in range(len(policy.W[i][j])):
                r = np.random.uniform()
                if r < p:
                    policy.W[i][j][k] = np.random.normal()
        for j in range(len(policy.B[i])):
            r = np.random.uniform()
            if r < p:
                policy.B[i][j] = np.random.normal()

def layer(num_in, num_out):
    w = np.random.normal(size = (num_in, num_out))
    b = np.random.normal(size = num_out)
    return w, b

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
