import random


class EpsilonGreedy:

    def __init__(self, bandits: list, epsilon=0.05):
        """
        Constructor for the epsilon-greedy algorithm

        :param bandits: A list of bandits to play
        :type bandits: list

        :param epsilon: Exploration vs exploitation tradeoff parameter
        :type epsilon: float
        """
        self.bandits = bandits
        self.epsilon = epsilon

        # Container for sample mean of each bandit
        self.sample_means = [0] * len(bandits)

        # Container for the number of plays of each bandit
        self.counts = [0] * len(bandits)

        # Internally track the number of steps run
        self.num_steps = 0

    def run(self, num_steps=10000):
        """ 
        Run the epsilon-greedy algorithm
        """
        total_rewards = 0
        for _ in range(num_steps):
            p = random.random()
            if p < self.epsilon:
                # Exploration. Select a random bandit.
                j = random.randint(0, len(self.bandits)-1)
            else:
                # Exploitation. Select the bandit with largest sample mean
                j = self._argmax(self.sample_means)

            bandit = self.bandits[j]
            reward = bandit.pull()
            total_rewards += reward

            # Update sample mean
            self._update_means(j, reward)

            self.num_steps += 1

        return total_rewards

    def recommend_bandit(self):
        """ Recommend the best bandit to play with the best reward """
        bandit_idx = self._argmax(self.counts)
        return self.counts[bandit_idx], bandit_idx

    def _argmax(self, l: list):
        """ Helper function for finding argmax of list l """
        max_i, max_v = 0, 0
        for i, v in enumerate(l):
            if v > max_v:
                max_i = i
                max_v = v

        return max_i

    def _update_means(self, idx: int, reward: float):
        """ Update the sample mean and counts of the i_th bandit """
        self.counts[idx] += 1
        self.sample_means[idx] = self.sample_means[idx] + 1 / \
            self.counts[idx] * (reward - self.sample_means[idx])


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


if __name__ == "__main__":
    # Do a 3 bandit experiment
    bandits = [SampleBandit(0.2), SampleBandit(0.6), SampleBandit(0.8)]
    algo = EpsilonGreedy(bandits)

    # Run 10 times
    rewards = algo.run(num_steps=10)
    count, i = algo.recommend_bandit()
    print("Total rewards is {} after {} steps".format(rewards, algo.num_steps))
    print("Bandit {} is played {} times after {} steps".format(
        i, count, algo.num_steps))
    print('')

    # Run 100 times
    rewards = algo.run(num_steps=90)
    count, i = algo.recommend_bandit()
    print("Total rewards is {} after {} steps".format(rewards, algo.num_steps))
    print("Bandit {} is played {} times after {} steps".format(
        i, count, algo.num_steps))
    print('')

    # Run 1000 times
    rewards = algo.run(num_steps=900)
    count, i = algo.recommend_bandit()
    print("Total rewards is {} after {} steps".format(rewards, algo.num_steps))
    print("Bandit {} is played {} times after {} steps".format(
        i, count, algo.num_steps))
    print('')

    # Run 10000 times
    rewards = algo.run(num_steps=9000)
    count, i = algo.recommend_bandit()
    print("Total rewards is {} after {} steps".format(rewards, algo.num_steps))
    print("Bandit {} is played {} times after {} steps".format(
        i, count, algo.num_steps))
    print('')
