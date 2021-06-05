from gridworld import gridworld, agents

def main():
    deterministic_simulation()
    monte_carlo_simulation()

def deterministic_simulation():
    # Some rudimentary testing
    env = gridworld.create_standard_grid()
    dp_agent = agents.DynamicProgrammingAgent(env)

    print("Running simulation on deterministic standard gridworld.")
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

    env = gridworld.create_costly_grid()
    dp_agent = agents.DynamicProgrammingAgent(env)
    print("Running simulation on costly gridworld")
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

def monte_carlo_simulation():
    # Run a monte carlo simulation
    env = gridworld.create_standard_grid()
    mc_agent = agents.MonteCarloAgent(env)

    print("Running monte carlo simulation on standard gridworld.")
    print("Initial values and policy:")
    mc_agent.evaluate_policy(env)
    mc_agent.print_values()
    mc_agent.print_policy()
    print("")

if __name__ == "__main__":
    main()

