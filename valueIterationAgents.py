# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    # question 1
    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            v_k = self.values.copy()

            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)

                    first_action = actions[0]
                    best_value = 0
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, first_action):
                        reward = self.mdp.getReward(state, first_action, next_state)
                        best_value += prob * (reward + self.discount * v_k[next_state])

                    for action in actions[1:]:
                        q_value = 0
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                            reward = self.mdp.getReward(state, action, next_state)
                            q_value += prob * (reward + self.discount * v_k[next_state])

                        if q_value > best_value:
                            best_value = q_value

                    self.values[state] = best_value

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    # question 1
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        q_value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            discounted_future = self.discount*self.values[next_state]
            q_value += prob*(reward + discounted_future)
        
        return q_value

    # question 1
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        actions = self.mdp.getPossibleActions(state)

        if len(actions) == 0:
            return None

        best_action = None
        best_q_value = None

        for action in actions:
            q_value = self.computeQValueFromValues(state, action)

            if best_q_value is None or q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
    
    # question 4 (extra credit)
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        priority_queue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            if next_state not in predecessors:
                                predecessors[next_state] = set()
                            predecessors[next_state].add(state)

                diff = abs(self.values[state] - self.bestQValue(state))
                priority_queue.push(state, -diff)

        for i in range(self.iterations):
            if priority_queue.isEmpty():
                break

            state = priority_queue.pop()

            if not self.mdp.isTerminal(state):
                self.values[state] = self.bestQValue(state)

            for pred in predecessors.get(state, set()):
                diff = abs(self.values[pred] - self.bestQValue(pred))
                if diff > self.theta:
                    priority_queue.update(pred, -diff)

    
    # helper method
    def bestQValue(self, state):
        actions = self.mdp.getPossibleActions(state)
        best_q = float('-inf')
        for action in actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_q:
                best_q = q_value
        return best_q