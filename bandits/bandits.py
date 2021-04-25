import random


class SampleBandit:
    """ A toy epsilon greedy bandit that rewards 1 if win and 0 if lose"""

    def __init__(self, win_rate=0.5):
        self.win_rate = win_rate

    def pull(self):
        """ Return a reward based on probability distribution """
        p = random.random()
        if p < self.win_rate:
            return 1
        else:
            return 0
