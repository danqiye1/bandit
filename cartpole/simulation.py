import gym
import matplotlib.pyplot as plt
from agents import ApproximateTDAgent
from pdb import set_trace as bp

def main():
    env = gym.make("CartPole-v1")
    agent = ApproximateTDAgent(env)
    deltas = agent.iterate_policy()
    plt.plot(deltas)
    plt.show()

    agent.play()
    
if __name__ == "__main__":
    main()