import numpy as np
import random
import matplotlib.pyplot as plt

w = [1, 1, 1, 1, 1, 1] #initialize weights
  
n = 100 # number of game
eta = 0.2 #[0,0.5]
expert = [0, 0, 0, 0, 0, 0] # prediction of each expert
prev_res = 0 # to determine win_streaks
win_streaks = 0
    
cum_loss_expert = [0, 0, 0, 0, 0, 0] #cumulative loss of expert
cum_loss_learner = 0

#for plotting
avg_cum_regret = []
learner_loss = []
expert1_loss = []
expert2_loss = []
expert3_loss = []
expert4_loss = []
expert5_loss = []
expert6_loss = []

def observations():
    global prev_res
    global win_streaks
    #features/observations
     # weather feature: sunny increase win chance
    sunny = random.choice([1, -1])
    
    # home vs away feature: home increase win chance
    home = random.choice([1, -1])
        
    # win_streak feature: higher win_streak increase win chance
    if ( prev_res == 1): 
        win_streaks += 1
    else: 
        win_streaks = 0 
    wins = win_streaks
    
    print("features sunny:{} home:{} wins:{}".format(sunny, home, wins))
    
    return sunny, home, wins

#experts make prediction
def expert_pred(sunny, home, wins):
    expert[0] = 1 # always perdicts win
    
    expert[1] = -1 # always predict loss
    
    if ((i+1) % 2 == 1): #lose odd, win even
        expert[2] = -1
    else:
        expert[2] = 1
    
    #expert4: 7/10 chance of predict win if sunny and 7/10 chance of predict loss if not sunny
    if (sunny==1):
        expert[3] = np.random.choice([1, -1], p = [7/10, 3/10])
    else:
        expert[3] = np.random.choice([1, -1], p = [3/10, 7/10])
    
    #expert5: 7/10 chance of predict win if sunny and home
    #         3/10 chance of predict win if not sunny and away
    #         6/10 chance of predict win if not sunny and home
    #         4/10 chance of predict win if sunny and away
    if (sunny == 1 and home == 1):
        expert[4] = np.random.choice([1, -1], p = [7/10, 3/10])
    elif(sunny == 1 and home == -1):
        expert[4] = np.random.choice([1, -1], p = [4/10, 6/10])
    elif(sunny == -1 and home == -1):
        expert[4] = np.random.choice([1, -1], p = [3/10, 7/10])
    elif(sunny == -1 and home == 1):
        expert[4] = np.random.choice([1, -1], p = [6/10, 4/10])
    
    #expert6: each win_streaks increase prediction of win by 5%
    win_chance = (wins * 5 + 50)/100
    expert[5] = np.random.choice([1, -1], p = [win_chance, 1 - win_chance])
    
    return expert

#three world class for the Tartan's sports team 
def real_world(sunny, home, wins, iters, a = "D"):
    #adversial
    if a == "A":
        ad_learner_pred = (w[0] * expert[0] + w[1] * expert[1] + w[2] * expert[2]
        + w[3] * expert[3] + w[4] * expert[4] + w[5] * expert[5])
        
        ret = -1 if np.sign(ad_learner_pred)>0 else 1

    #stochastic
    if a == "S":
        #win 10% more when sunny
        #win 2% more each win strikes
        #win 13% more when home
        win_rate = (wins * 1 + 12 * home + 7 * sunny + 50)/100
        if (win_rate >= 1):
            win_rate = 1
        ret = np.random.choice([1, -1], p = [win_rate, 1 - win_rate])
        
    #deterministic
    if a == "D":
        if (home == 1):
            ret = 1
        else:
            ret = -1
            
    
    return ret

#WPA
for i in range(n):#at each time step t
    
    print("round:{}".format(i+1))
    
    sunny, home, wins = observations() # observe
    expert = expert_pred(sunny, home, wins) # expert predict
    
    print("expert: {}".format(expert))
    '''
    #RWMA
    #learner's prediction = random selection based on weights
    normalized_weights = [i/sum(w) for i in w]
    expert_num = np.random.choice(6, 1, p = normalized_weights)
    y = expert[expert_num[0]]
    
    '''
    #WMA
    #learner makes prediction based on experts
    learner_pred = (w[0] * expert[0] + w[1] * expert[1] + w[2] * expert[2]
        + w[3] * expert[3] + w[4] * expert[4] + w[5] * expert[5])
    y = 1 if np.sign(learner_pred)>0 else -1
    
    
    #outcome/true label is revealed
    yt = real_world(sunny, home, wins, i) 
    prev_res = yt  
    
    print("weights:{}".format(w))
    print("learner:{}, true label:{}".format(y, yt))
    #adjust experts weights
    for j in range(6):
        if (expert[j] != yt):
            w[j] = w[j] * (1 - eta) 
            #print(w)
    
    #calculate cumulative loss of exps and learner
    for j in range(6):
        if (expert[j] != yt): 
            cum_loss_expert[j] += 1
            
    if(y != yt):
        cum_loss_learner += 1
    #print("cum_loss_expert1: {}".format(cum_loss_expert[2]))
    
    #calculate regret
    R = cum_loss_learner - min(cum_loss_expert)
    #print("min_loss_expert: {}".format(min(cum_loss_expert)))
    #save regre/loss values

    avg_cum_regret.append(R/(i+1))
    learner_loss.append(cum_loss_learner)
    expert1_loss.append(cum_loss_expert[0])
    expert2_loss.append(cum_loss_expert[1])
    expert3_loss.append(cum_loss_expert[2])
    expert4_loss.append(cum_loss_expert[3])
    expert5_loss.append(cum_loss_expert[4])
    expert6_loss.append(cum_loss_expert[5])
    #print("   ")
    
plt.plot(np.arange(1,101), avg_cum_regret)
plt.xlabel('Time steps')
plt.ylabel('Average cumulative regret')
plt.show()

plt.plot(np.arange(1,101), learner_loss,label='learner loss')
plt.plot(np.arange(1,101), expert1_loss,label='expert1 loss')
plt.plot(np.arange(1,101), expert2_loss,label='expert2 loss')
plt.plot(np.arange(1,101), expert3_loss,label='expert3 loss')
plt.plot(np.arange(1,101), expert4_loss,label='expert4 loss')
plt.plot(np.arange(1,101), expert5_loss,label='expert5 loss')
plt.plot(np.arange(1,101), expert6_loss,label='expert6 loss')
plt.legend(loc = 'best')
plt.xlabel('Time steps')
plt.ylabel('Cumulative loss')
plt.show()
