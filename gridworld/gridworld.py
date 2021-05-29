class GridWorld:
    """ GridWorld is a deterministic 2D environment. """
    def __init__(self, rewards, actions):
        """ 
        Initialize the state-rewards mapping and the initial state of the agent 
        
        :param rewards: A mapping of state to rewards
        :type rewards: dict

        :param actions: A mapping of state to actions
        :type actions: dict

        :param state: Initial state of the agent
        :type state: tuple
        """
        self.rewards = rewards
        self.actions = actions

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

def create_standard_grid():
    """ 
    Create a standard 4 x 3 grid world with terminal states (2,3), (1,3)
    The transitions to the terminal states yield 1 and -1 rewards respectively.
    State (1,1) is inaccessible
    """
    rewards = {}

    # Default 4x3 grid world with terminal states (2,3), (1,3)
    # With rewards 1 and -1 respectively
    # State (1,1) is inaccessible
    for i in range(3):
        for j in range(4):
            rewards[(i,j)] = 0
    
    rewards[(2,3)] = 1
    rewards[(1,3)] = -1

    actions = {
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

    env = GridWorld(rewards, actions)

    return env

def create_costly_grid(cost=-0.1):
    """ 
    Create a costly 4 x 3 grid world with terminal states (2,3), (1,3)
    The transitions to the terminal states yield 1 and -1 rewards respectively.
    State (1,1) is inaccessible.
    Every step taken will cost a negative reward
    """
    env = create_standard_grid()
    for s in env.rewards:
        if not env.is_terminal(s):
            env.rewards[s] = cost

    return env