import numpy as np

class Agent(object):

    def __init__(self):
        self.Q = {}
        #self.epsilon = epsilon
        self.reward_log = []

    def policy(self, s, actions, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(len(actions))
        else:
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []
        print(self.reward_log)

    def log(self, reward):
        self.reward_log.append(reward)
