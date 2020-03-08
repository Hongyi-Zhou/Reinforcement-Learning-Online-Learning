import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from policy import *
from game import *

'''
#University Website Latency Dataset
mat = scipy.io.loadmat('../data/univLatencies.mat')
data = mat['univ_latencies']
game = gameLookupTable(data, isLoss = True)
'''

#Planner Dataset
mat = scipy.io.loadmat('../data/plannerPerformance.mat')
data = mat['planner_performance']
game = gameLookupTable(data, isLoss = True)


#game = gameAdverserial()
 
policies = [policyEXP3(), policyUCB()]
policy_names = ['policyEXP3', 'policyUCB']#,'policyConstant']

fig = plt.figure()

for k in range(len(policies)):
    game.resetGame()
    if type(policies[k]) == policyUCB:
        ini = game.initilization()
        policies[k].init(game.nbActions,ini)
    reward,action,regret = game.play(policies[k])
    #reward,action,regret, upperconfidence = game.play(policies[k])
    #plot action count
#    count1 = 0
#    count2 = 0
#    a1count = np.zeros(action.shape)
#    a2count = np.zeros(action.shape)
#    for i in range(len(action)):
#        if action[i] == 0:
#            count1 += 1
#        else:
#            count2 += 1
#        a1count[i] = count1
#        a2count[i] = count2
#        
#    plt.figure()
#    plt.plot(a1count,label=policy_names[k] + ' action1')
#    plt.plot(a2count,label=policy_names[k] + ' action2')
#    plt.xlabel('trials')
#    plt.ylabel('action')
#    plt.legend()
    
    print("{} Reward {:.2f}".format(policy_names[k],reward.sum()))
    plt.figure(0)
    plt.plot(regret,label=policy_names[k])
    plt.xlabel('trials')
    plt.ylabel('regret')
    

#    
#    plt.figure(2)
#    plt.plot(upperconfidence[:,0],label=policy_names[k] + ' action1')
#    plt.plot(upperconfidence[:,1],label=policy_names[k] + ' action2')
#    plt.xlabel('trials')
#    plt.ylabel('upper confidence')
plt.legend()
plt.show()

