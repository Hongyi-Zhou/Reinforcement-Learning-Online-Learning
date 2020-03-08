#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:42:53 2019

@author: superzhy
"""

import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
 
## Load a CSV file
def load(files):
    fh = open("pcdata.txt", 'w')
    for filename in files:
        f = open(filename, "r")
        #f.next()
        for line in f:
            d = line.split()
            if len(d)>10:
                for i in range(3):
                    fh.write(d[i]+ " " )
                
                for i in range(5,len(d)-1):
                    fh.write(d[i]+ " " )
                    
                if (int(d[4]) == 1004):
                    label = 0 #veg
                elif (int(d[4]) == 1100):
                    label = 1 #wire
                elif (int(d[4]) == 1103):
                    label = 2 #pole
                elif (int(d[4]) == 1200):
                    label = 3 #ground
                elif (int(d[4]) == 1400):
                    label = 4 #facade
                    
                fh.write(str(label) + "\n" )    

#split data into training, test set
def split_data(data, ratio):
    trainSize = int(len(data) * ratio)
    np.random.shuffle(data)
    
    train_x = data[:trainSize,:]
    test_x = data[trainSize:,:]
    train_y = train_x[:,-1]
    test_y = test_x[:,-1]
    train_x = np.delete(train_x, -1, axis = 1)
    test_x = np.delete(test_x, -1, axis = 1)
    
    return train_x, train_y, test_x, test_y

#fit SVM model
def fit(x, y , max_iter = 5000, eta = 0.01):
    x = np.insert(x, 0, 1, axis = 1) #add intercept
        
    classes = np.unique(y) # num of classes
    classes = classes.astype(int)

    weights_multi = np.zeros((len(classes), x.shape[1])) #store weights for different classes
    
    M = []
    
    t_mistakes = 0
    
    for c in classes:
        # One vs. All classification
        binary_label = np.where(y == c, 1, -1)# generate label
        
        weights = np.zeros(x.shape[1])#multi_class weights
        for i in range(max_iter):
            
            m = np.random.randint(x.shape[0])# uniform stochastic sample
            
            y_hat = np.sign(weights @ x[m,:])# prediction
            
            if y_hat != binary_label[m]:# save number of mistakes
                t_mistakes += 1 
            
            if (binary_label[m]*(weights @ x[m,:])) < 1:#weights update
                weights += eta * binary_label[m] * x[m,:]
            else:
                weights = weights
            
            if c == 4:
                M.append(t_mistakes/(i+1.) )  #record the average number of mistakes

        weights_multi[c,:] = weights
    
    return weights_multi, M, classes

def predict(weights_multi, x):
    x = np.insert(x, 0, 1, axis = 1) #add intercept
    classesProb = np.dot(x, weights_multi.T)
    predictions = classesProb.argmax(axis = 1)# predictions
    predictions = predictions.astype(int)
    
    return predictions

#concatenate two files
load(['oakland_part3_am_rf.node_features', 'oakland_part3_an_rf.node_features'])
#load data
x = loadtxt("pcdata.txt", comments="#", delimiter=" ", unpack=False)

train_x, train_y, test_x, test_y = split_data(x , 0.8)
'''
#tune hyperparameter
tune = np.zeros((20,3))
e = np.linspace(0.006,0.012,4)
ite = np.linspace(1,9,5).astype(int)

for i in range(len(ite)):
    for j in range(len(e)):
        start_time = time.time()
        print(ite[i]*10000)
        weights, M, classes = fit(train_x[:,3:12], train_y, max_iter = ite[i]*10000, eta = e[j])
        end_time = time.time()
        #print("---%s seconds"%(end_time-start_time))
        
        predictions_train = predict(weights, train_x[:,3:12])
        print ("Train Accuracy:", str(100*np.mean(predictions_train == train_y)) + "%")
        
        start_time = time.time()
        predictions_test = predict(weights, test_x[:,3:12])
        print ("Test Accuracy:", str(100*np.mean(predictions_test == test_y)) + "%")
        end_time = time.time()
        tune[5*j+i][0] = ite[i]*10000#save iterations
        tune[5*j+i][1] = e[j]#save learning rate
        tune[5*j+i][2] = 100*np.mean(predictions_test == test_y)#save accuracy

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(tune[:,0],tune[:,1],tune[:,2], s = 0.125)
plt.title('tune parameters')
plt.show()
'''
  
start_time = time.time()
weights, M, classes = fit(train_x[:,3:12], train_y, max_iter = 100000, eta = 0.01)
end_time = time.time()
print("---%s seconds"%(end_time-start_time))

predictions_train = predict(weights, train_x[:,3:12])
print ("Train Accuracy:", str(100*np.mean(predictions_train == train_y)) + "%")

start_time = time.time()
predictions_test = predict(weights, test_x[:,3:12])
print ("Test Accuracy:", str(100*np.mean(predictions_test == test_y)) + "%")
end_time = time.time()
print("---%s seconds"%(end_time-start_time))
'''    
for i in classes:
    a = np.where(test_y == i)
    b = predictions_test[a[0]]
    print("Test Accuracy for class {} of {} elements".format(i,len(a[0])), str(100*np.mean(b == i)) + "%")
'''
#for confusion matrix
c = np.zeros((len(classes),len(classes)))
for i in classes:
    a = np.where(test_y == i)
    b = predictions_test[a[0]]
    for j in classes:       
        c[i][j] = len(np.where(b == j)[0])
    

start_time = time.time()
predictions_all = predict(weights, x[:,3:12])
end_time = time.time()
print("---%s seconds"%(end_time-start_time))

#visualize point clouds

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0],x[:,1],x[:,2], s = 0.125, c = predictions_all)
plt.title('Classified Point Cloud')
plt.show()


'''
plt.plot(M)
plt.xlabel('t')
plt.ylabel('Mt/t (number of mistakes till step t)')
plt.show()
'''










