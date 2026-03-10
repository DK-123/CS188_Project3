# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.0 # changed this from 0.2 to 0.0 so the agent moves deterministically
    return answerDiscount, answerNoise

# Prefer the close exit (+1), risking the cliff (-10)
def question3a():
    answerDiscount = 0.1 # has to be 0 < d <= 0.5 to prefer closer exit so any value w/ this condition works
    answerNoise = 0.0
    answerLivingReward = -1.0
    return answerDiscount, answerNoise, answerLivingReward

# Prefer the close exit (+1), but avoiding the cliff (-10)
def question3b():
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = -1.0 # agent wants to exit asap
    return answerDiscount, answerNoise, answerLivingReward

#Prefer the distant exit (+10), risking the cliff (-10)
def question3c():
    answerDiscount = 0.8
    answerNoise = 0.0 
    answerLivingReward = -0.1
    return answerDiscount, answerNoise, answerLivingReward

# Prefer the distant exit (+10), avoiding the cliff (-10)
def question3d():
    answerDiscount = 0.8
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

# Avoid both exits and the cliff (so an episode should never terminate)
def question3e():
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 10.0 # has to be positive and large enough to outweigh exits
    return answerDiscount, answerNoise, answerLivingReward

def question7():
    # answerEpsilon = 0
    # answerLearningRate = 1
    # return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
