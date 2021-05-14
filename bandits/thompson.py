import numpy as np
from bandits import BernoulliBandit, NormalBandit

class BernoulliThompson:

    def __init__(self, bandits: list):
        """
        Constructor for Bernoulli Thompson Sampler.
        This Thompson Sampler is used if we believe the bandits are drawing from Bernoulli distribution

        :param bandits: A list of bandits to play
        :type bandits: list
        """
        K = len(bandits)
        self.bandits = bandits
        
        # Initialize beta distribution parameters for all bandits
        self.alpha = [1.0] * K
        self.beta = [1.0] * K

        # Keep track of bandit rewards and pull rate
        self.rewards = [0.0] * K 
        self.pull_rate = [0] * K

    def update(self, x: float, k: int):
        """ 
        Function to update beta distribution of bandit k

        :param x: Update variable. This is often the reward value.
        :param k: Index of bandit to update
        """
        # Update beta distribution parameters
        self.alpha[k] += x
        self.beta[k] += 1 - x

        # Update experiment results
        self.rewards[k] += x
        self.pull_rate[k] += 1

    def sample(self, k: int):
        """ Sample the posterior of bandit k """
        return np.random.beta(self.alpha[k], self.beta[k])

    def run(self, num_steps=10000):
        """ Run thompson sampling algorithm """
        for _ in range(num_steps):
            # Sample the distribution and get the argmax
            j = np.argmax([self.sample(k) for k in range(len(self.bandits))])

            # Play the selected bandit
            reward = self.bandits[j].pull()

            # Update the beta distribution parameters for the jth bandit
            self.update(reward, j)

class NormalThompson:

    def __init__(self, bandits: list, tau=1.0):
        """
        Constructor for Normal Thompson Sampler.
        This Thompson Sampler is used if we believe the bandits are drawing from normal distribution,
        with known precision tau and unknown mean mu

        :param bandits: A list of bandits to play
        :type bandits: list

        :param tau: Precision of the likelihood estimate for all bandits
        :type tau: float
        """
        # Randomly initialize the posterior parameters
        self.bandits = bandits
        K = len(self.bandits)
        self.means = np.random.randn(K)
        self.precisions = np.ones(K)
        self.tau = tau

        # Keep track of experiment runs
        self.pull_rate = np.zeros(K)
        self.rewards = np.zeros(K)

    def run(self, num_steps=10000):
        """ Run the Thompson Sampling Algorithm """
        for _ in range(num_steps):
            # Estimate sample mean from distribution
            j = np.argmax([self.sample(k) for k in range(len(self.bandits))])

            # Play the selected bandit
            reward = self.bandits[j].pull()

            # Update the beta distribution parameters for the jth bandit
            self.update(reward, j)

    def update(self, x: float, k: int):
        """ 
        Function to update beta distribution of bandit k

        :param x: Update variable. This is often the reward value.
        :param k: Index of bandit to update
        """
        self.precisions[k] += self.tau
        self.means[k] = 1/self.precisions[k] * (self.tau * x + self.means[k] * self.precisions[k])

        self.pull_rate[k] += 1
        self.rewards[k] += x

    def sample(self, k: int):
        """ 
        Draw a posterior sample of sample mean (mu) from normal distribution 
        
        :param k: index of the bandit which we want to sample
        """
        return np.random.normal(self.means[k], 1/self.precisions[k])

if __name__ == "__main__":
    # Do a 3 bandit experiment
    bandits = [BernoulliBandit(0.2), BernoulliBandit(0.6), BernoulliBandit(0.8)]
    K = len(bandits)
    algo = BernoulliThompson(bandits)

    # Run 10 times
    algo.run(num_steps=10)
    print("Experiment ran for {} steps".format(np.sum(algo.pull_rate)))
    for k in range(K):
        print("Bandit {} is played {} times with estimated {} win rate.".format(
            k, algo.pull_rate[k], algo.rewards[k]/algo.pull_rate[k]
        ))
    print("")

    # Run 100 times
    algo.run(num_steps=90)
    print("Experiment ran for {} steps".format(np.sum(algo.pull_rate)))
    for k in range(K):
        print("Bandit {} is played {} times with estimated {} win rate.".format(
            k, algo.pull_rate[k], algo.rewards[k]/algo.pull_rate[k]
        ))
    print("")

    # Run 1000 times
    algo.run(num_steps=900)
    print("Experiment ran for {} steps".format(np.sum(algo.pull_rate)))
    for k in range(K):
        print("Bandit {} is played {} times with estimated {} win rate.".format(
            k, algo.pull_rate[k], algo.rewards[k]/algo.pull_rate[k]
        ))
    print("")

    # Run 10000 times
    algo.run(num_steps=9000)
    print("Experiment ran for {} steps".format(np.sum(algo.pull_rate)))
    for k in range(K):
        print("Bandit {} is played {} times with estimated {} win rate.".format(
            k, algo.pull_rate[k], algo.rewards[k]/algo.pull_rate[k]
        ))
    print("")

    # Do a 3 bandit experiment for Gaussian Bandits
    bandits = [NormalBandit(0.2), NormalBandit(-0.4), BernoulliBandit(-1.0)]
    K = len(bandits)
    algo = NormalThompson(bandits)

    # Run 10 times
    algo.run(num_steps=10)
    print("Experiment ran for {} steps".format(np.sum(algo.pull_rate)))
    for k in range(K):
        print("Bandit {} is played {} times with estimated {} win rate.".format(
            k, algo.pull_rate[k], algo.rewards[k]/algo.pull_rate[k]
        ))
    print("")

    # Run 100 times
    algo.run(num_steps=90)
    print("Experiment ran for {} steps".format(np.sum(algo.pull_rate)))
    for k in range(K):
        print("Bandit {} is played {} times with estimated {} win rate.".format(
            k, algo.pull_rate[k], algo.rewards[k]/algo.pull_rate[k]
        ))
    print("")

    # Run 1000 times
    algo.run(num_steps=900)
    print("Experiment ran for {} steps".format(np.sum(algo.pull_rate)))
    for k in range(K):
        print("Bandit {} is played {} times with estimated {} win rate.".format(
            k, algo.pull_rate[k], algo.rewards[k]/algo.pull_rate[k]
        ))
    print("")

    # Run 10000 times
    algo.run(num_steps=9000)
    print("Experiment ran for {} steps".format(np.sum(algo.pull_rate)))
    for k in range(K):
        print("Bandit {} is played {} times with estimated {} win rate.".format(
            k, algo.pull_rate[k], algo.rewards[k]/algo.pull_rate[k]
        ))
    print("")