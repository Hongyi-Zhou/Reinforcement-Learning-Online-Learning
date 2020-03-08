import numpy as np

class Policy:
    """
    DO NOT MODIFY
    """
    def init(self, nbActions):
        self.nbActions = nbActions
    def decision(self):
        pass
    def getReward(self,reward):
        pass

class policyRandom(Policy):
    """
    DO NOT MODIFY
    """
    def decision(self):
        return np.random.randint(0,self.nbActions,dtype=np.int)
    def getReward(self,reward):
        pass

class policyConstant(Policy):
    """
    DO NOT MODIFY
    """
    def init(self,nbActions):
        self.chosenAction = np.random.randint(0,nbActions,dtype=np.int)
    def decision(self):
        return self.chosenAction
    def getReward(self,reward):
        pass

class policyGWM(Policy):
    def init(self, nbActions):
        self.nbActions = nbActions
        self.weights = np.ones(self.nbActions)
        self.action = 0;
        self.t = 0;
    
    def decision(self):
        self.t += 1 # number of rounds
        normalized_weights = self.weights/sum(self.weights)
        self.action = np.random.choice(self.nbActions, 1, p = normalized_weights) # prediction
        return self.action
    
    def getReward(self,reward):
        loss = np.zeros(self.nbActions)
        loss[self.action] = 1 - reward
        eta = np.sqrt(np.log(self.nbActions)/self.t)
        self.weights = self.weights*np.exp(-eta*loss)
        
class policyEXP3(Policy):
    def init(self, nbActions):
        self.nbActions = nbActions
        self.weights = np.ones(self.nbActions)
        self.action = 0;
        self.t = 0;

    def decision(self):
        self.t += 1 # number of rounds
        normalized_weights = self.weights/sum(self.weights)
        self.action = np.random.choice(self.nbActions, 1, p = normalized_weights) # prediction
        return self.action
    
    def getReward(self,reward):
        loss = np.zeros(self.nbActions)
        loss[self.action] = 1 - reward
        normalized_weights = self.weights/sum(self.weights)
        loss = loss/normalized_weights
        eta = np.sqrt(np.log(self.nbActions)/(self.t*self.nbActions))
        self.weights = self.weights*np.exp(-eta*loss)

class policyUCB(Policy):
    def init(self, nbActions, ini):
        self.ini = ini
        self.nbActions = nbActions
        self.S = np.zeros(self.nbActions)
        self.C = np.zeros(self.nbActions)
        self.alpha = 0.2
        self.t = 0;
        self.action = 0;
        for i in range(self.nbActions):
            self.C[i] = 1
        self.S = ini
        
    def decision(self):
        self.t += 1
        cal = self.S/self.C + np.sqrt(self.alpha*np.log(self.t)/(2*self.C))
        c = np.sqrt(self.alpha*np.log(self.t)/(2*self.C))
        self.action = np.argmax(cal)
        return self.action, c
    
    def getReward(self,reward):
        self.S[self.action] += reward
        self.C[self.action] += 1
        
