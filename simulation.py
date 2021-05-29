from gridworld import gridworld, agents

def main():
    deterministic_simulation()

def deterministic_simulation():
    # Some rudimentary testing
    env = gridworld.create_standard_grid()
    dp_agent = agents.DynamicProgrammingAgent(env)

    print("Initial values and policy:")
    dp_agent.evaluate_policy(env)
    dp_agent.print_values()
    dp_agent.print_policy()
    print("")

    print("Optimised policy:")
    dp_agent.iterate_policy(env)
    dp_agent.print_values()
    dp_agent.print_policy()
    print("")

if __name__ == "__main__":
    main()

