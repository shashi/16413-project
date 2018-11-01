import numpy as np
from search_classes import *

def eucl_dist(a, b):
    """Returns the euclidean distance between a and b."""
    return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))


class MDP(object):
    """Simple MDP class."""
    def __init__(self, S, A, T, R, gamma):
        """Define MDP"""
        self.S = S # Set of states
        self.A = A # Set of actions
        self.T = T # Transition probabilities: T[s][a][s']
        self.R = R # Rewards: R[s][a][s']
        self.gamma = gamma # Discount factor


def value_iteration(mdp, epsilon=1e-3):
    states, actions, gamma = mdp.S, mdp.A, mdp.gamma
    V_iteration_new = dict([(s, 0) for s in mdp.S])

    iter = 0
    expanded_states = []

    while True:
        # Initialize
        V_iteration = V_iteration_new.copy()
        delta = 0
        iter = iter + 1

        # Calculate Bellman's Equation
        for state in states:
            expanded_states.append(state)
            V_iteration_temp = []
            for action in actions[state]:
                V_iteration_temp_temp = 0
                for s_new in mdp.T[state][action]:
                    try:
                        r = mdp.R[state][action][s_new]
                    except:
                        r = 0
                    p = mdp.T[state][action][s_new]
                    V_iteration_temp_temp += p * (r + mdp.gamma *  V_iteration[s_new])
                V_iteration_temp.append(V_iteration_temp_temp)
            V_iteration_new[state] = max(V_iteration_temp)
            delta = max(delta, abs(V_iteration[state] - V_iteration_new[state]))

        if delta < epsilon * (1 - gamma) / gamma:
            print("The value iteration needed " + str(iter) + " iterations to converge, " + 
                  "and explored " + str(len(expanded_states)) + " states")
            return V_iteration, expanded_states


def extract_policy(mdp, V):
    Pi = dict()

    for s in mdp.S:
        actions, values = [], []
        for a in mdp.T[s].keys():
            actions.append(a)
            tmp = 0
            for s_new, prob in mdp.T[s][a].items():
                if s_new not in mdp.R[s][a].keys():
                    tmp = tmp + prob * mdp.gamma * V[s_new]
                else:
                    tmp = tmp + prob * (mdp.R[s][a][s_new] + mdp.gamma * V[s_new])
            values.append(tmp)
        Pi[s] = actions[values.index(max(values))]

    return Pi


def best_first_search(problem, f):
    """Returns a solution path."""
    q = PriorityQueue(f=f)
    q.append(SearchNode(problem.start))
    expanded = {problem.start}
    max_q = 1
    while q:
        # get max length of queue
        max_q = max(max_q, len(q))

        # Get element from the queue
        new_start = q.pop()
        expanded.add(new_start.state)

        # Check if goal found
        if problem.test_goal(new_start.state):
            return Path(new_start), len(expanded), max_q

        # Get the goal_nodes for the new node
        goal_nodes = problem.expand_node(new_start)

        # Check if already visited
        goal_nodes = [goal_node for goal_node in goal_nodes if goal_node not in expanded]

        for goal_node in goal_nodes:
            # Check if node already expanded
            if goal_node.state not in expanded:

                # check if goal node is already somewhere in the queue
                if goal_node in q:
                    # check which goal node should be kept in the queue
                    previous_goal_node = q[goal_node]
                    # if new goal is better
                    if previous_goal_node.cost > goal_node.cost:
                        del q[previous_goal_node]
                        q.append(goal_node)
                else: # if goal node is not in queue
                    q.append(goal_node)