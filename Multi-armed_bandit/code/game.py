# Credit to 16831 TA
import numpy as np
from policy import *

class Game:
    nbActions = 0
    totalRounds = 0
    N = 0
    tabR = np.array([])
    cum_best_action_reward = 0

    def __init__(self):
        return
    def play(self,policy):
        #print(policy)
        if type(policy) != policyUCB:
            policy.init(self.nbActions)
            
        reward = np.zeros(self.totalRounds)
        action = np.zeros(self.totalRounds,dtype=np.int)
        regret = np.zeros(self.totalRounds)
        
        if type(policy) == policyUCB:
            upperconfidence = np.zeros((self.totalRounds, self.nbActions))

        for t in range(self.totalRounds):
            
            if type(policy) == policyUCB:
                action[t], a = policy.decision()
                #upperconfidence[t,:] = a
            else:
                action[t] = policy.decision()
            #print(action[t])
            
            reward[t] = self.reward(action[t])
            regret[t] = self.cumulativeRewardBestActionHindsight() - sum(reward)
            #print(self.N, reward[t], regret[t])
            policy.getReward(reward[t])
            self.N += 1
            #if (self.N % 1000 == 0): print(self.N)

        #return reward, action, regret, upperconfidence
        return reward, action, regret
        
    def reward(self,a):
        #print(self.tabR.shape)
        return self.tabR[a, self.N]
    def resetGame(self):
        self.N = 0
        self.cum_best_action_reward = 0
    def cumulativeRewardBestActionHindsight(self):
        self.cum_best_action_reward += np.max(self.tabR[:, self.N])
        return self.cum_best_action_reward
    def initilization(self): # for UCB initialization
        return self.tabR[:,0] 

class gameConstant(Game):
    """
    DO NOT MODIFY
    """
    def __init__(self):
        super().__init__()
        self.nbActions = 2
        self.totalRounds = 1000
        self.tabR = np.ones((2,1000))
        self.tabR[0] *= 0.8
        self.tabR[1] *= 0.2
        self.N = 0

class gameGaussian(Game):
    def __init__(self,nbActions,totalRound):
        super().__init__()
        self.nbActions = nbActions
        self.totalRounds = totalRound
        self.tabR = np.zeros((self.nbActions,self.totalRounds))
        for i in range(self.nbActions):
            mu = np.random.randint(0, 1001) / 1000.0
            sigma = np.random.randint(0, 1001) / 1000.0
            self.tabR[i,:] = np.random.normal(mu, sigma, self.totalRounds) #gaussian distribution 
        self.tabR[self.tabR < 0] = 0
        self.tabR[self.tabR > 1] = 1
        self.N = 0
        #print(self.tabR[:,5])
        #print(self.tabR[:,8])
        print("done")

class gameAdverserial(Game):
    def __init__(self):
        super().__init__()
        self.nbActions = 2
        self.totalRounds = 2000
        self.tabR = np.zeros((self.nbActions,self.totalRounds))
        for i in range(self.totalRounds):
            if i%2==0:
                self.tabR[0,i] = 1
                self.tabR[1,i] = 0.2
            else:
                self.tabR[1,i] = 1
                self.tabR[0,i] = 0.2
        self.N = 0

class gameLookupTable(Game):
    def __init__(self,tabInput,isLoss):
        super().__init__()
        self.N = 0
        self.tabR = tabInput
        self.isLoss = isLoss
        self.nbActions = tabInput.shape[0]
        self.totalRounds = tabInput.shape[1]
        
        
