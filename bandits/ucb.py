import numpy as np
from bandits import BernoulliBandit


class UpperConfidenceBound:

    def __init__(self, bandits: list):
        """
        Constructor for the upper confidence bound algorithm

        :param bandits: A list of bandits to play
        :type bandits: list
        """
        self.bandits = bandits

        # Internally track the total rewards
        self.total_rewards = 0

        self.sample_means = np.zeros(len(bandits))

        # Container for the number of plays of each bandit
        self.counts = np.zeros(len(bandits))

        # Internally track the number of steps run
        self.num_steps = 0

        # Initialize the sample mean by playing each bandit once
        for idx, bandit in enumerate(self.bandits):
            reward = bandit.pull()
            self.total_rewards += reward

            # Update sample mean
            self._update_means(idx, reward)

            self.num_steps += 1

    def run(self, num_steps=10000):
        """ 
        Run the algorithm
        """

        for _ in range(num_steps):
            # Choose the bandit with the largest sample mean estimate
            j = np.argmax(self.sample_means - np.sqrt(
                2 * np.log(self.num_steps)/self.counts
            ))

            bandit = self.bandits[j]
            reward = bandit.pull()
            self.total_rewards += reward

            # Update sample mean
            self._update_means(j, reward)

            self.num_steps += 1

        return self.total_rewards

    def recommend_bandit(self):
        """ Recommend the best bandit to play with the best reward """
        bandit_idx = np.argmax(self.sample_means)
        return self.counts[bandit_idx], bandit_idx

    def _update_means(self, idx: int, reward: float):
        """ Update the sample mean and counts of the i_th bandit """
        self.counts[idx] += 1
        self.sample_means[idx] = self.sample_means[idx] + 1 / \
            self.counts[idx] * (reward - self.sample_means[idx])


if __name__ == "__main__":
    # Do a 3 bandit experiment
    bandits = [BernoulliBandit(0.2), BernoulliBandit(0.6), BernoulliBandit(0.8)]
    algo = UpperConfidenceBound(bandits)

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
