import numpy as np
from gridworld import GridWorld
from pdb import set_trace as bp

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
                    print(s_prime, state)
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
                print(max_change)
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

if __name__ == "__main__":
    # Some rudimentary testing
    env = GridWorld()
    dp_agent = DynamicProgrammingAgent(env)
    dp_agent.print_values()
    dp_agent.evaluate_policy(env)
    dp_agent.print_values()
    dp_agent.print_policy()
    