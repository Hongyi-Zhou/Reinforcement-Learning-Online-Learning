import numpy as np
import cvxopt as cvx

import gridworld
import rl

def irl_lp(policy, T_probs, discount, R_max, l1):
  """
  Solves the linear program formulation for finite discrete state IRL.

  Inputs:
    policy: np.ndarray
      Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
    T_probs: np.ndarray
      nS x nA x nS matrix where:
      T_probs[s][a] is a probability distribution over states of transitioning
      from state s using action a.
      Can be generated using env.generateTransitionMatrices.
    gamma: float
      Discount factor, must be in range [0, 1)
    R_max: float
      Maximum reward allowed.
    l1: float
      L1 regularization penalty.

  Output:
    np.ndarray
    R: Array of rewards for each state.
  """

  T_probs = np.asarray(T_probs) # states, actions, states
  nS, nA, _ = T_probs.shape

  ###### C matrix
  c = np.zeros((3*nS,1)) #R
  c[nS:2*nS] = -1  #q
  c[2*nS:] = l1  #j
  
  ######### B matrix
  
  b = np.zeros((nS*(nA-1)*2 + 2*nS + 2*nS, 1))
  b[-2*nS:] = R_max #last 2 n terms are Rmax
  
  ############## A matrix
  
  def dotproduct(s, a):
      eqn1 = T_probs[s][int(policy[s])] - T_probs[s][a]
      eqn2 = np.linalg.inv(np.eye(nS) - discount * T_probs[:,int(policy[s]),:])
      #print(eqn1.shape, eqn2.shape)
      return -eqn1 @ eqn2
  
  a11 = np.zeros((0,nS))
  for s in range(nS):
      for a in range(nA):
          if int(policy[s]) == a: 
             continue
          #print(dotproduct(s,a).reshape(1,-1).shape) 
          a11 = np.row_stack((a11, dotproduct(s,a).reshape(1,-1)))
          
  #print(nS, nA)
  a21 = a11
      
  a31 = -np.eye(nS)
  a41 = np.eye(nS)
  a51 = -np.eye(nS)
  a61 = np.eye(nS)
  
  #print(a11.shape, a31.shape)
  
  a12 = np.zeros((0,nS))
  for s in range(nS):
      for a in range(nA):
          if int(policy[s]) == a: 
             continue
          #print(dotproduct(s,a).reshape(1,-1).shape) 
          a12 = np.row_stack((a12, np.eye(1, nS, s)))
  
  a22 = np.zeros((nS*(nA-1), nS))
  a32 = np.zeros((nS, nS))
  a42 = np.zeros((nS, nS))
  a52 = np.zeros((nS, nS))
  a62 = np.zeros((nS, nS))
  
  a13 = np.zeros((nS*(nA-1), nS))
  a23 = np.zeros((nS*(nA-1), nS))
  a33 = -np.eye(nS)
  a43 = -np.eye(nS)
  a53 = np.zeros((nS, nS))
  a63 = np.zeros((nS, nS))
  
  ac1 = np.vstack([a11, a21, a31, a41, a51, a61])
  ac2 = np.vstack([a12, a22, a32, a42, a52, a62])
  ac3 = np.vstack([a13, a23, a33, a43, a53, a63])
  
  a = np.hstack([ac1, ac2, ac3])

  #solve linear programming problem
  c = cvx.matrix(c)
  G = cvx.matrix(a)
  h = cvx.matrix(b)
  sol = cvx.solvers.lp(c, G, h)
#
  R = np.asarray(sol["x"][:nS]).squeeze()

  return R

if __name__ == "__main__":
  env = gridworld.GridWorld(map_name='4x4')

  gamma = 0.9
  Vs, n_iter = rl.value_iteration(env, gamma)
  policy = rl.policy_from_value_function(env, Vs, gamma)

  T = env.generateTransitionMatrices()

  R_max = 1
  l1 = 1e-3

  R = irl_lp(policy, T, gamma, R_max, l1)
  r = np.around(R.reshape((4,4)),4)
  #print(r)

  env_irl = gridworld.GridWorld(map_name='4x4', R=R)
  Vs_irl, n_iter_irl = rl.value_iteration(env_irl, gamma)
  policy_irl = rl.policy_from_value_function(env_irl, Vs_irl, gamma)
  pp = policy_irl.reshape((4,4))
  ppp = policy.reshape((4,4))
  