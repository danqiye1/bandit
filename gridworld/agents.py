import numpy as np

class DynamicProgrammingAgent:

    def __init__(self, env, delta=1e-3, gamma=0.9, transition_probs=None):
        """ 
        A dynamic programming agent for solving GridWorld 
        
        :param delta: Threshold for stopping policy iteration
        :type delta: float

        :param gamma: Discount factor for Bellman Equation
        :type gamma: float

        :param env: An interface between the agent and the environment
        """
        # Initialize hyper-parameters
        self.delta = delta
        self.gamma = gamma

        # Initialize action space
        self.actions = ["up", "down", "left", "right"]

        # Initialize current state
        self.state = (0,0)

        # Keep a counter on the number of iterations
        self.steps = 0

        # Initialize transition probabilities
        # This is p(s_prime, r | s, a) in the Bellman Equation
        # Note that although p(s_prime, r | s, a) is distribution of environment parameters
        # it is the agent's responsibility to keep track of it
        if not transition_probs:
            self.transition_prob = {}
            self.rewards = {}
            self.num_rows = 0
            self.num_columns = 0
            for state in env.get_states():

                self.num_rows = max(self.num_rows, state[0] + 1)
                self.num_columns = max(self.num_columns, state[1] + 1)

                for action in self.actions:
                    r , s_prime = env.move(state, action)
                    if s_prime != state:
                        # A transition occurred. Set the transition probablility to 1
                        # In more complex environments this step should be replaced by a learning function
                        self.transition_prob[(s_prime, state, action)] = 1
                        self.rewards[(s_prime, state, action)] = r
        else:
            self.transition_prob = transition_probs
    

        # Initialize a policy pie(a|s)
        self.policy = {
            (0,0): "down",
            (0,1): "right",
            (0,2): "down",
            (0,3): "down",
            (1,0): "down",
            (1,2): "down",
            (2,0): "right",
            (2,1): "right",
            (2,2): "right",
        }

        # Initialize the value function for all possible states in environment
        self.V = {}
        for state in env.get_states():
            self.V[state] = 0

    def evaluate_policy(self, env):
        """ Perform policy evaluation using dynamic programming """
        while True:
            max_change = 0
            for s in env.get_states():
                if not env.is_terminal(s):
                    old_v = self.V[s]
                    new_v = 0
                    for action in self.actions:
                        
                        # Derive the pi(a|s), which is deterministic in this case
                        if action == self.policy.get(s):
                            action_prob = 1
                        else:
                            action_prob = 0

                        for s_prime in env.get_states():
                            reward = self.rewards.get((s_prime, s, action), 0)
                            # Bellman update equation
                            new_v += action_prob * self.transition_prob.get((s_prime, s, action), 0) * (reward + self.gamma * self.V[s_prime])
                        
                    self.V[s] = new_v
                    max_change = max(max_change, np.abs(old_v-new_v))


            self.steps += 1
            
            # Break when converged
            if max_change <= self.delta:
                break

    def iterate_policy(self, env):
        """ Perform policy iteration using dynamic programming """
        while True:
            converged = True # Tracks if policy has converged
            self.evaluate_policy(env)
            
            for s in env.get_states():
                if s in self.policy:
                    old_action = self.policy[s]
                    new_action = None
                    best_value = float('-inf')

                    # Loop through all possible actions in action space to find the best action
                    # This is like a complicated argmax
                    for a in self.actions:
                        v = 0
                        for s_prime in env.get_states():
                            r = self.rewards.get((s_prime, s, a), 0)
                            v += self.transition_prob.get((s_prime, s, a), 0) * (r + self.gamma * self.V[s_prime])

                        if v > best_value:
                            best_value = v
                            new_action = a

                    self.policy[s] = new_action
                    if new_action != old_action:
                        converged = False

            if converged:
                break

    def print_values(self):
        """ Print the current values of states in gridworld """
        border = "-"
        for j in range(self.num_columns):
            border += "-------"
        print(border)

        for i in range(self.num_rows):
            row = "| "
            for j in range(self.num_columns):
                row += "%.2f" % self.V[(i,j)] + " | "
            print(row)

        print(border)


    def print_policy(self):
        """ Print the current policy for this agent on gridworld """
        border = "-"
        for j in range(self.num_columns):
            border += "----"
        print(border)

        # Map for consistent spacing during print
        actions_map = {
            "": " ",
            "up": "U",
            "down": "D",
            "left": "L",
            "right": "R"
        }

        for i in range(self.num_rows):
            row = "| "
            for j in range(self.num_columns):
                row += actions_map[self.policy.get((i,j), "")] + " | "
            print(row)

        print(border)


class MonteCarloAgent:

    def __init__(self, env, gamma=0.9, policy=None):
        self.gamma = gamma

        # Initialize policy
        if policy:
            self.policy = policy
        else:
            # Define a random policy if policy is not given
            self.policy = {
                (0,0): "down",
                (0,1): "left",
                (0,2): "right",
                (0,3): "left",
                (1,0): "down",
                (1,2): "up",
                (2,0): "right",
                (2,1): "right",
                (2,2): "right",
            }

        self.action_space = ["up", "down", "left", "right"]

        # Record num_rows and num_columns for printing values and policy
        self.num_rows = 0
        self.num_columns = 0

        # Initialize value function
        self.V = {}
        self.ret = {}
        self.Q = {}
        self.retQ = {}
        for s in env.get_states():
            self.num_rows = max(self.num_rows, s[0] + 1)
            self.num_columns = max(self.num_columns, s[1] + 1)

            for a in self.action_space:
                if not env.is_terminal(s):
                    self.ret[s] = []
                    self.retQ[(s,a)] = []
                else:
                    self.V[s] = 0
                    self.Q[(s,a)] = 0           


    def play(self, env, max_steps=20, explore_start=False):
        """
        Play an episode of max_steps in environment. 
        If explore_start is True, the first action is random and not according to self.policy

        :param env: environment for agent to interact with

        :param max_steps: maximum number of steps for this episode.
        :type max_steps: int

        :param explore_start: Parameter to choose if we are doing exploration start, where first action is not according to self.policy
        :type explore_start: bool
        """
        # Randomly initialize the start state
        s_idx = np.random.choice(len(env.get_states()))
        s = list(env.get_states())[s_idx]
        states = [s]
        rewards = [0]
        actions = []
        steps = 0
        
        while not env.is_terminal(s) and (steps < max_steps):

            if steps == 0 and explore_start:
                # Exploration start
                action = np.random.choice(self.action_space)
            else:
                # Sample pi(a|s) to get a valid action
                action = self.policy[s]

            # Move the agent and obtain a reward
            r, s_prime = env.move(s, action)

            # Record the states, rewards, and actions of the episode
            states.append(s_prime)
            rewards.append(r)
            actions.append(action)

            # Update state
            s = s_prime

            # Increment steps
            steps += 1

        return states, rewards, actions

    def evaluate_policy(self, env, num_episodes=100):
        for _ in range(num_episodes):
            # Play one episode of the game
            states, rewards, _  = self.play(env)
            G = 0
            for t in range(len(states) - 2, -1, -1):
                s = states[t]
                r = rewards[t+1]
                G = r + self.gamma * G

                # Implement a every-visit monte-carlo
                self.ret[s].append(G)
                self.V[s] = np.mean(self.ret[s])

    def iterate_policy_es(self, env, num_episodes=100):
        """ 
        Monte Carlo Policy Iteration algorithm with exploring starts.
        Randomly select an initial state and action to start, and follow self.policy after.
        """
        for _ in range(num_episodes):
            # Play one episode of the game
            states, rewards, actions = self.play(env, explore_start=True)
            G = 0
            for t in range(len(states) - 2, -1, -1):
                s = states[t]
                r = rewards[t+1]
                a = actions[t]

                G = r + self.gamma * G

                # Implement every-visit monte-carlo
                self.retQ[(s,a)].append(G)
                self.Q[(s,a)] = np.mean(self.retQ[(s,a)])

                # Update the best action to our policy
                Q = []
                for action in self.action_space:
                    Q.append(self.Q.get((s,action), float('-inf')))
                self.policy[s] = self.action_space[np.argmax(Q)]

    def iterate_policy(self, env, epsilon=0.1, num_episodes=1000):
        """
        Monte Carlo policy iteration without exploring starts
        """
        for _ in range(num_episodes):
            # Play one episode of the game
            states, rewards, actions = self.play(env)
            G = 0
            for t in range(len(states) - 2, -1, -1):
                s = states[t]
                r = rewards[t+1]
                a = actions[t]

                G = r + self.gamma * G

                # Implement every-visit monte-carlo
                self.retQ[(s,a)].append(G)
                self.Q[(s,a)] = np.mean(self.retQ[(s,a)])

                # Update the policy based on epsilon greedy
                p = np.random.random()
                if p < epsilon:
                    # Return random action
                    self.policy[s] = np.random.choice(self.action_space)
                else:
                    # Return best action
                    Q = []
                    for action in self.action_space:
                        Q.append(self.Q.get((s,action), float('-inf')))
                    self.policy[s] = self.action_space[np.argmax(Q)]

    def print_values(self):
        """ Print the current values of states in gridworld """
        border = "-"
        for j in range(self.num_columns):
            border += "-------"
        print(border)

        for i in range(self.num_rows):
            row = "| "
            for j in range(self.num_columns):
                row += "%.2f" % self.V[(i,j)] + " | "
            print(row)

        print(border)

    def print_policy(self):
        """ Print the current policy for this agent on gridworld """
        border = "-"
        for j in range(self.num_columns):
            border += "----"
        print(border)

        # Map for consistent spacing during print
        actions_map = {
            "": " ",
            "up": "U",
            "down": "D",
            "left": "L",
            "right": "R"
        }

        for i in range(self.num_rows):
            row = "| "
            for j in range(self.num_columns):
                row += actions_map[self.policy.get((i,j), "")] + " | "
            print(row)

        print(border)

class TemporalDifferenceAgent:

    def __init__(self, env, start_state=(0,0), initial_policy=None, action_space=None):
        
        if initial_policy:
            self.policy = initial_policy
        else:
            # Define a random policy if policy is not given
            self.policy = {
                (0,0): "down",
                (0,1): "left",
                (0,2): "right",
                (0,3): "left",
                (1,0): "down",
                (1,2): "up",
                (2,0): "right",
                (2,1): "right",
                (2,2): "right",
            }

        # Initialize action_space
        if action_space:
            self.action_space = action_space
        else:
            self.action_space = ["up", "down", "left", "right"]

        # Initialize V(s) and Q(s,a)
        self.num_rows = 0
        self.num_columns = 0
        self.V = {}
        self.Q = {}
        for s in env.get_states():
            self.num_rows = max(self.num_rows, s[0] + 1)
            self.num_columns = max(self.num_columns, s[1] + 1)
            self.V[s] = 0
            for a in self.action_space:
                self.Q[(s,a)] = np.random.random()

        # Initialize state of agent
        self.start_state = start_state
        
    def evaluate_policy(self, env, delta=1e-9, alpha=0.1, gamma=0.9):
        """ 
        TD Prediction Code: Learns value function by traversing Gridworld
        
        :param env: Gridworld environment
        :param delta: Difference in V for convergence
        :param alpha: learning rate of TD Prediction.

        :return deltas: list of differences to track convergence
        """
        # List of deltas to track convergence
        deltas = []

        while True:
            s = self.start_state
            diff = 0
            while not env.is_terminal(s):
                a = self.policy[s]
                r, s_prime = env.move(s, a)

                # Update V
                old_v = self.V[s]
                self.V[s] = self.V[s] + alpha * (r + gamma * self.V[s_prime] - self.V[s])
                diff = max(diff, np.abs(self.V[s] - old_v))
                s = s_prime
                
            deltas.append(diff)
            # Converged
            if diff < delta:
                break

        return deltas

    def iterate_policy(self, env, algo='sarsa', delta=1e-3, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Policy iteration implementation with both SARSA and Q Learning

        :param env: Gridworld environment
        :param algo: The algorithm to use. Can either be sarsa or q_learning
        :param delta: Threshold for convergence
        :param alpha: Learning rate
        :param gamma: Bellman equation discount factor
        :param epsilon: Probablity of choosing a random action
        """
        # List of deltas to track convergence
        deltas = []

        while True:
            s = self.start_state
            diff = 0
            while not env.is_terminal(s):
                a = self._select_action(s, epsilon)
                r, s_prime = env.move(s, a)

                # Update Q(s,a)
                old_Q = self.Q[(s,a)]
                if algo == 'sarsa':
                    a_prime = self._select_action(s_prime, epsilon)
                    self.Q[(s,a)] = old_Q + alpha * (r + gamma * self.Q[(s_prime, a_prime)] - old_Q)
                elif algo == 'q_learning':
                    Q = []
                    for action in self.action_space:
                        Q.append(self.Q.get((s_prime ,action), float('-inf')))
                    self.Q[(s,a)] = old_Q + alpha * (r + gamma * max(Q) - old_Q)
                else:
                    raise RuntimeError("Invalid Algorithm {} chosen".format(algo))
                diff = max(diff, np.abs(self.Q[(s,a)] - old_Q))

                s = s_prime

            deltas.append(diff)
            if diff < delta:
                # Converged
                break

        # Update optimal policy
        for s in env.get_states():
            if not env.is_terminal(s):
                Q = []
                for action in self.action_space:
                    Q.append(self.Q.get((s ,action), float('-inf')))
                self.policy[s] = self.action_space[np.argmax(Q)]
                self.V[s] = max(Q)

        return deltas

    def _select_action(self, state, epsilon):
        """ Select the next action from Q(s,a) using epsilon greedy algorithm """
        p = np.random.random()
        if p < epsilon:
            # Return a random action
            a = np.random.choice(self.action_space)
        else:
            # Return best action
            Q = []
            for action in self.action_space:
                Q.append(self.Q.get((state,action), float('-inf')))
            a = self.action_space[np.argmax(Q)]

        return a

    def print_values(self):
        """ Print the current values of states in gridworld """
        border = "-"
        for j in range(self.num_columns):
            border += "-------"
        print(border)

        for i in range(self.num_rows):
            row = "| "
            for j in range(self.num_columns):
                row += "%.2f" % self.V[(i,j)] + " | "
            print(row)

        print(border)

    def print_policy(self):
        """ Print the current policy for this agent on gridworld """
        border = "-"
        for j in range(self.num_columns):
            border += "----"
        print(border)

        # Map for consistent spacing during print
        actions_map = {
            "": " ",
            "up": "U",
            "down": "D",
            "left": "L",
            "right": "R"
        }

        for i in range(self.num_rows):
            row = "| "
            for j in range(self.num_columns):
                row += actions_map[self.policy.get((i,j), "")] + " | "
            print(row)

        print(border)