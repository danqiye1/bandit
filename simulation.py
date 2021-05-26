from gridworld import gridworld, agents

def main():
    deterministic_simulation()

def deterministic_simulation():
    # Some rudimentary testing
    env = gridworld.GridWorld()
    dp_agent = agents.DynamicProgrammingAgent(env)
    dp_agent.print_values()
    dp_agent.evaluate_policy(env)
    dp_agent.print_values()
    dp_agent.print_policy()

if __name__ == "__main__":
    main()

