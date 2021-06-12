from gridworld import gridworld, agents
import matplotlib.pyplot as plt

def main():
    deterministic_simulation()
    monte_carlo_simulation()
    td_simulation()

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
    mc_agent.print_policy()
    mc_agent.print_values()
    print("")

    print("Optimizing Policy with Exploration Start:")
    mc_agent.iterate_policy_es(env)
    mc_agent.print_policy()
    mc_agent.evaluate_policy(env)
    mc_agent.print_values()
    print("")

    print("Optimizing Policy with Epsilon Greedy:")
    # Reinitialize agent
    mc_agent = agents.MonteCarloAgent(env)
    mc_agent.iterate_policy(env)
    mc_agent.print_policy()
    mc_agent.evaluate_policy(env)
    mc_agent.print_values()
    print("")

def td_simulation():
    # Run temporal difference simulation
    env = gridworld.create_standard_grid()
    td_agent = agents.TemporalDifferenceAgent(env)

    print("Running temporal difference simulation on deterministic standard gridworld.")
    print("Initial values and policy:")
    deltas = td_agent.evaluate_policy(env)
    td_agent.print_values()
    td_agent.print_policy()
    print("")

    plt.plot(deltas)
    plt.show()

    print("Optimizing Policy with SARSA:")
    deltas = td_agent.iterate_policy(env, algo='sarsa')
    td_agent.print_policy()
    td_agent.print_values()
    print("")

    plt.plot(deltas)
    plt.show()

    print("Optimizing Policy with Q-Learning:")
    # Reinitialize agent
    td_agent = agents.TemporalDifferenceAgent(env)
    deltas = td_agent.iterate_policy(env, algo='q_learning')
    td_agent.print_policy()
    td_agent.print_values()
    print("")

    plt.plot(deltas)
    plt.show()

if __name__ == "__main__":
    main()

