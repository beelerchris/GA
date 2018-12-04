import numpy as np
import gym
import gym.spaces

class Policy():
    def __init__(self, shape, hidden_units, num_actions, a_bound, game):
        self.shape = shape
        self.game = game
        self.hidden_units = hidden_units
        self.num_actions = num_actions
        self.a_bound = a_bound
        self.win = 0
        self.W = []
        self.B = []

    def gen_random(self, seed=None):
        np.random.seed(seed)
        self.W = []
        self.B = []
        w, b = layer(self.shape, self.hidden_units[0])
        self.W.append(w)
        self.B.append(b)

        for i in range(1, len(self.hidden_units)):
            w, b = layer(self.hidden_units[i-1], self.hidden_units[i])
            self.W.append(w)
            self.B.append(b)

        w, b = layer(self.hidden_units[-1], self.num_actions)
        self.W.append(w)
        self.B.append(b)

    def evaluate(self, state):
        Y = np.tanh(np.matmul(state, self.W[0]) + self.B[0])

        for i in range(1, len(self.W)):
            Y = np.tanh(np.matmul(Y, self.W[i]) + self.B[i])

        Y = (Y + 1.0) / 2.0

        return Y * (self.a_bound[1] - self.a_bound[0]) + self.a_bound[0]

def layer(num_in, num_out):
    w = np.random.normal(scale = 1.0/(num_in*num_out), size = (num_in, num_out))
    b = np.random.normal(scale = 1.0/(num_in*num_out), size = num_out)
    return w, b

