class GridWorld:
    """ GridWorld is a deterministic 2D environment. """
    def __init__(self, rewards=None, actions=None):
        """ 
        Initialize the state-rewards mapping and the initial state of the agent 
        
        :param rewards: A mapping of state to rewards
        :type rewards: dict

        :param actions: A mapping of state to actions
        :type actions: dict

        :param state: Initial state of the agent
        :type state: tuple
        """

        if rewards and actions:
            self.rewards = rewards
            self.actions = actions
        elif rewards:
            # Only rewards is provided and not actions
            raise RuntimeError("No state to actions mapping provided in constructor!")
        elif actions:
            # Only actions is provided and not rewards
            raise RuntimeError("No state to rewards mapping provided in constructor!")
        else:
            self.rewards = {}
            self.actions = {}

            # Default 4x3 grid world with terminal states (2,3), (1,3)
            # With rewards 1 and -1 respectively
            # State (1,1) is inaccessible
            for i in range(3):
                for j in range(4):
                    self.rewards[(i,j)] = 0

            self.rewards[(2,3)] = 1
            self.rewards[(1,3)] = -1

            self.actions = {
                (0,0): ("down", "right"),
                (0,1): ("left", "right"),
                (0,2): ("down", "left", "right"),
                (0,3): ("down", "left"),
                (1,0): ("up", "down"),
                (1,2): ("up", "down", "right"),
                (2,0): ("up", "right"),
                (2,1): ("left", "right"),
                (2,2): ("up", "left", "right"),
            }

    def move(self, state, action):
        """ Given an agent action, return the reward and update it's position/state """
        allowed_actions = self.actions.get(state, ())

        if action in allowed_actions:
            if action == "up":
                s_prime = (state[0] - 1, state[1])
            elif action == "down":
                s_prime = (state[0] + 1, state[1])
            elif action == "left":
                s_prime = (state[0], state[1] - 1)
            elif action == "right":
                s_prime = (state[0], state[1] + 1)
            
            return (self.rewards[s_prime], s_prime)
        else:
            return (0, state)

    def is_terminal(self, state):
        """ Determine is agent is in terminal state """
        return state not in self.actions

    def get_states(self):
        """ Return a set of all possible states """
        # In gridworld states are the keys of self.rewards
        return self.rewards.keys()