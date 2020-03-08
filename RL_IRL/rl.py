import numpy as np
import gridworld 

def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
  """
  This implements value iteration for learning a policy given an environment.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Tolerance used for stopping criterion based on convergence.
      If the values are changing by less than tol, you should exit.

  Output:
    (numpy.ndarray, iteration)
    value_function:  Optimal value function
    iteration: number of iterations it took to converge.
  """
  value = np.zeros(env.nS)
  prev_value = np.zeros(env.nS)
  #actions = np.zeros(env.nA)
  iterations = 0
  while (iterations < max_iterations):
#      env.reset()
      iterations += 1
      record = np.zeros((env.nS, env.nA))
      for state in range(env.nS):
          for action in range(env.nA):
              for items in env.P[state][action]:
                  record[state][action] += items[0]*(items[2] + gamma * value[items[1]])
              #print(env.P[state][action], state, action)
          prev_value[state] = value[state]        
          value[state] = max(record[state])
      err = abs(prev_value-value)
      #print(value.reshape((4,4)))
#      
#      if(iterations %10 ==0):
#          print("iter: {}".format(iterations))
          
      if all(err < tol):
          print("Value iterations done. Iter: {}".format(iterations))
          break

  return value, iterations

def policy_from_value_function(env, value_function, gamma):
  """
  This generates a policy given a value function.
  Useful for generating a policy given an optimal value function from value
  iteration.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    value_function: numpy.ndarray
      Optimal value function array of length nS
    gamma: float
      Discount factor, must be in range [0, 1)

  Output:
    numpy.ndarray
    policy: Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
  """

  policy = np.zeros(env.nS)
  for state in range(env.nS):
      val_action = np.zeros(env.nA)
      for action in range(env.nA):
          for items in env.P[state][action]:
              val_action[action] += items[0]*(items[2] + gamma * value_function[items[1]])
#              if (state==0):
#                  print(val_action)
#      if (state==0):
#          print(value_function)
#          print(val_action, items[2])
      policy[state] = np.argmax(val_action)
    
  return policy

def evaluate_policy(env, policy, gamma, tol):
    values = np.zeros(env.nS)
    
    while(1):
        prev_values = values
        values = np.zeros(env.nS)
        for state in range(env.nS):
            for items in env.P[state][policy[state]]:
                values[state] += items[0]*(items[2] + gamma * prev_values[items[1]])
        err = abs(prev_values - values)
        
        if all(err < tol):
          break
        
    return values

def improve_policy(env, value, gamma):
    return policy_from_value_function(env, value, gamma)


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-10):
  """
  This implements policy iteration for learning a policy given an environment.

  You should potentially implement two functions "evaluate_policy" and 
  "improve_policy" which are called as subroutines for this.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Tolerance used for stopping criterion based on convergence.
      If the values are changing by less than tol, you should exit.

  Output:
    (numpy.ndarray, iteration)
    value_function:  Optimal value function
    iteration: number of iterations it took to converge.
  """
  iterations = 0
  policy = np.random.choice(env.nA, size=(env.nS)) 
      
  while(iterations < max_iterations):
    iterations += 1
    prev_policy = policy
    #policy evaluation
    values = evaluate_policy(env, policy, gamma, tol)
    #policy improvement
    policy = improve_policy(env, values, gamma)
  
    if all(policy == prev_policy):
        print("Policy iterations done: Iter: {}".format(iterations))
        print(values.reshape((8,8)))
        break  
  
  return values, iterations

def td_zero(env, gamma, policy, alpha):
  """
  This implements TD(0) for calculating the value function given a policy.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: numpy.ndarray
      Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
    alpha: float
      Learning rate/step size for the temporal difference update.

  Output:
    numpy.ndarray
    value_function:  Policy value function
  """
  #tol = 1e-10
  value_function = np.zeros(env.nS)#np.random.random(env.nS)
  epochs = 0
  while (epochs < 10000):
      epochs += 1
      state = env.reset()
      done = 0
      #prev_val = value_function
      
      while (not done):
          action = policy[state]
          n_state, reward, done, prob = env.step(action)
          value_function[state] += alpha * (reward + gamma * value_function[n_state] - value_function[state])
          state = n_state
          
#      if (epochs%1000 == 0):
#          print("function after {} epochs: {}\n".format(epochs, value_function.reshape((8,8))))
##      err = abs(value_function - prev_val)
#          
#      if all(err < tol):
#          print("Td zeros done. Epochs: {}".format(epochs))
#          break

  return value_function

def n_step_td(env, gamma, policy, alpha, n):
  """
  This implements n-step TD for calculating the value function given a policy.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: numpy.ndarray
      Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
    n: int
      Number of future steps for calculating the return from a state.
    alpha: float
      Learning rate/step size for the temporal difference update.

  Output:
    numpy.ndarray
    value_function:  Policy value function
  """
  #pseudocode from https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=2ahUKEwjGoYeYq93lAhWhTt8KHVofA8EQFjABegQIBBAH&url=https%3A%2F%2Fiu.instructure.com%2Fcourses%2F1600154%2Ffiles%2F68944465%2Fdownload%3Fverifier%3DAdHv5cN8E7vbOZ6yifP8swa3h57nzrt7RgUFwJBT%26wrap%3D1&usg=AOvVaw3fjKDgX5BCv2Zcf2P1AgMB
  #initialize value function arbitrarily
  value_function = np.zeros(env.nS)
  epochs = 0
  while (epochs < 10000):
      epochs += 1
      done = 0
      t = 0
      T = np.inf
      
      state = env.reset() #store initial state
      
      action_hist = []
      reward_hist = [0]
      state_hist = [state]
      
      while (1):
          
          if t < T:
              action = policy[state_hist[t]] #sample action
              n_state, reward, done, prob = env.step(action)
              #store reward, state, action
              action_hist.append(action)
              reward_hist.append(reward)
              state_hist.append(n_state)

              if (done): 
                  T = t + 1

          tau = t - n + 1 #tau is the time whose state's estimate is being updated
          
          if tau >= 0 :
              G = sum([gamma**(i-tau-1) * reward_hist[i] for i in range(tau+1, min(tau+n,T)+1)])
              if tau + n < T:
                  G += gamma**n * value_function[state_hist[tau+n]]
              value_function[state_hist[tau]] += alpha * (G - value_function[state_hist[tau]]) 
            
          if tau == T-1 : 
                break;
          
          t += 1
          
#      if (epochs%1000 == 0):
#          print("function after {} epochs: {}\n".format(epochs, value_function.reshape((4,4))))

  return value_function

if __name__ == "__main__":
  env = gridworld.GridWorld(map_name='8x8')

  gamma = 0.9
  alpha = 0.05
  n = 10
  
  V_vi, n_iter = value_iteration(env, gamma)
  policy = policy_from_value_function(env, V_vi, gamma)
  v = V_vi.reshape((8,8))
  p = policy.reshape((8,8))
#
#  Policy Iteration
#  V_pi, n_iter = policy_iteration(env, gamma)
#  policy = policy_from_value_function(env, V_pi, gamma)
#  v = V_pi.reshape((8,8))
#  p = policy.reshape((8,8))

#  TD-zero
#  V_td = td_zero(env, gamma, policy, alpha)
#  vv = V_td.reshape((4,4))

# n-step TD
  V_ntd = n_step_td(env, gamma, policy, alpha, n)
  vvv = V_ntd.reshape((8,8))