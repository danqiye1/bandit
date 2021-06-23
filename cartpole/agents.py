import numpy as np
from sklearn.kernel_approximation import RBFSampler

class ApproximateTDAgent:

    def __init__(self, env, num_episodes=10000):
        """
        Constructor for Temporal Difference Agent using function approximation.
        The function approximator used is a linear regression with RBF Kernel: y = np.dot(W, ph(x))

        :param env: OpenAI Gym environment to interface with
        :param num_episodes: number of episodes to play to bootstrap phi(x) and W
        """

        # Interface with the environment
        self.env = env

        # Initialize featurizer function phi(x)
        self.featurizer = RBFSampler()
        samples = []
        done = False
        for n in range(num_episodes):
            print("Running initial exploration episode: {}".format(n))
            s = self.env.reset()
            while not done:
                # Play the game randomly
                a = self.env.action_space.sample()
                x = self._vectorize(s,a)
                samples.append(x)
                s, _, done, _ = self.env.step(a)
        
        self.featurizer.fit(samples)
        self.W = np.zeros(self.featurizer.n_components)

    def _vectorize(self, s, a):
        """ 
        Helper function to vectorize state s and action a.
        
        :param s: state
        :type s: tuple
        :param a: action
        :type a: int
        """
        s = np.array(s)
        # One-hot encoding of actions
        a_vector = np.zeros(self.env.action_space.n)
        a_vector[a] = 1
        return np.concatenate((s,a_vector))

    def iterate_policy(self, alpha=0.1, gamma=0.9, epsilon=0.3, num_episodes=1000):
        """ Implementation of Q learning on the environment """

        deltas = []
        for n in range(num_episodes):
            print("Iterating episode {}".format(n))
            s = self.env.reset()
            done = False
            max_diff=float("-inf")
            while not done:
                a = self._select_action(s, epsilon)
                s_prime, r, done, _ = self.env.step(a)

                if done:
                    y = r
                else:
                    y = r + gamma * np.max(self.predict(s_prime))

                phi_x = self.featurizer.transform([self._vectorize(s, a)])[0]
                diff = y - np.dot(self.W, phi_x)
                self.W = self.W + alpha * diff * phi_x
                max_diff = max(max_diff, diff)
                s = s_prime

            deltas.append(max_diff)

        return deltas

    def _select_action(self, state, epsilon):
        """ 
        Helper function to choose between the explore-exploit dilemma 
        This is actually the pi(a|s) function
        """
        p = np.random.random()
        if p <= epsilon:
            selected_action = self.env.action_space.sample()
        else:
            Q_values = self.predict(state)
            selected_action = np.argmax(Q_values)
        
        return selected_action

    def predict(self, state):
        """
        Predict the Q values for all actions of input state

        :param state: state for which Q is predicted
        """
        # Calculate estimate of Q from dot(W, phi(x))
        # This is a linear regression model
        Q_values = []
        for action in range(self.env.action_space.n):
            x = self._vectorize(state, action)
            x = self.featurizer.transform([x])[0]
            Q_values.append(np.dot(self.W, x))

        return Q_values

    def play(self):
        """
        Play the agent according to current policy
        """
        done = False
        s = self.env.reset()
        total_rewards = 0
        while not done:
            # Always play according to policy
            a = self._select_action(s, epsilon=0.0)
            s, r, done, info = self.env.step(a)
            self.env.render()
            total_rewards += r

        return total_rewards
                
