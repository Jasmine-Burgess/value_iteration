def value_iteration(probs:np.array, rewards: np.array, states: list = None, actions: list = None, horizon: int = 1000, discount: float = 1) -> np.array:
    """
    Calculates the value function and optimal policy for a given problem

    Arguments
    ----
    probs: the transition probabilities between states given an action is taken. A 3-dimensional np.array with the first dimension corresponding
    to the states you are currently in, the second dimension to the different actions, and the third dimension to the states you will transition into.
    The elements of probs are probabilities. 
    
    rewards: the (immediate) rewards for each action in each state. A 2-dimensional np.array with the rows corresponding to the different
    states and columns to different actions. The elements of rewards are the immediate rewards.

    states:

    actions:

    horizon:

    Output
    ---


    
    """

    #Checking inputs for validity and returning appropriate error messages
    
    if np.shape(probs)[0] != np.shape(probs)[2] :
        return("Invalid transition probability matrix")
    if np.shape(rewards)[0] != np.shape(probs)[0] or np.shape(rewards)[1] != np.shape(probs)[1]:
        return("Reward and transition matrix are not compatiable")
    if len(states) not in [0, np.shape(probs)[0]] :
        return("State list not compatiable with reward and transition matrices")
    if len(actions) not in [0, np.shape(rewards)[1]] :
        return("Action list not compatiable with reward and transition matrices")
    for i in range(np.shape(probs)[0]):
       for j in range(np.shape(probs)[1]):
           if ( abs( sum(probs[i, j]) - 1) > 0.001):
               return("The transition probabilities for state " + str(i) + "and action" + str(j) + "do not sum to 1.") 


    state_num = np.shape(probs)[0]
    action_num = np.shape(probs)[1]

    
    val = np.zeros(state_num)
    val_act = np.zeros(action_num)
    k = 0
    for k in range(horizon):
        k = k + 1
        for s in range(state_num):
            for a in range(action_num):
                val_act[a] = rewards[s, a] + discount*sum(probs[s, a]*val)
            val[s] = max(val_act)
    pi = np.zeros(state_num)
    for s in range(state_num):
        for a in range(action_num):
            val_act[a] = rewards[s, a] + discount*sum(probs[s, a]*val)
        pi[s] = np.argmax(val_act)
    return(pi, val)