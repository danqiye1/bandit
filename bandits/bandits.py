import random
import numpy as np


class BernoulliBandit:
    """ A toy bernoulli bandit that rewards 1 if win and 0 if lose"""

    def __init__(self, win_rate=0.5):
        self.win_rate = win_rate

    def pull(self):
        """ Return a reward based on probability distribution """
        p = random.random()
        if p < self.win_rate:
            return 1
        else:
            return 0

class NormalBandit:
    """ A toy gaussian bandit that rewards 1 if win and 0 if lose based on a normal distribution"""

    def __init__(self, mu, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def pull(self):
        """ Return a reward based on the probability distribution """
        return np.random.normal(self.mu, self.sigma)