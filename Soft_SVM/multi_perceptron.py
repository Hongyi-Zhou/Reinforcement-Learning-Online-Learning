import numpy as np 
import matplotlib.pyplot as plt


#make prediction:
def multi_perceptron_predict(x_t, W_t):
    '''
    Multi-class perceptron prediction

    input: 
        x_t: d-dim feature vector
        W_t: the current weight matrix in dim (k X d):

    Output: 
        hat{y}_t: the prediction your perceptron algorithm would make
    '''

    #TODO: implement the prediction procedure:
    Wx = np.dot(W_t,x_t)
    
    hat_y = int(np.argmax(Wx)) #replace this line with your implemention 

    return hat_y

#update Weight matrix:
def multi_perceptron_update(x_t, y_t, W_t):
    '''
    Multi-class perceptron update procedure:
    
    input: 
            x_t: d-dim feature vector
            y_t: label, integer from [0,1,2..,k-2,k-1]
            W_t: the current Weight Matrix in dim (k X d)
    
    Output:
            W_tp1: the updated weighted matrix W_{t+1}.
    '''
   
    #TODO: implement the update rule to compute W_{t+1}:    \
    U_t = np.zeros_like(W_t)
    y_hat = multi_perceptron_predict(x_t, W_t)

    if (y_hat != y_t):
        U_t[y_t, :] = x_t.T
        U_t[y_hat, :] = -x_t.T
        
    W_tp1 = W_t + U_t  #replace this line with your implementation


    return W_tp1


def online_perceptron(X, Y):
    '''
    We put every pieces we implemented above together in this function.
    This function simulates the online learning procedure.  
    (you do not need to implement anything here)

    input:
        X: N x d, where N is the number of examples, and d is the dimension of the feature.
        Y: N-dim vector.
            --Multi-Class: Y[i] is the label for example X[i]. Y[i] is an integer from [0,1.,..k-2,k-1] (k classes).
            --Binary: Y[i] is 0 or 1. 
    
    output: 
        M: a list: M[t] is the average number of mistakes we make up till and including time step t. Note t starts from zero.  
           you should expect M[t] decays as t increases....
           you should expect M[-1] to be around ~0.2 for the mnist dataset we provided. 

        W: final predictor.
            --Binary: W is a d-dim vector, 
            --Multi-Class: W is a k X d matrix. 
    '''

    d = X.shape[1]   #feature dimension.
    k = np.unique(Y).shape[0] #num of unique labels.  k=2: binary, k>2: multi-class
    M = []

    t_mistaks = 0
    #Initialization for W:
    W = np.zeros((k, d)) if k>2 else np.zeros(d) 
    
    #we scan example one by one:
    for t in range(X.shape[0]):
        x_t = X[t]    #nature reveals x_t. 
        hat_y_t = multi_perceptron_predict(x_t, W) #we make prediction
        y_t = Y[t]   #nature reveals y_t after we make prediction.

        if y_t != hat_y_t: 
            t_mistaks = t_mistaks + 1
        
        W = multi_perceptron_update(x_t, y_t, W) #perceptron update. 

        M.append(t_mistaks/(t+1.) )  #record the average number of mistakes.

    return M, W



if __name__ == "__main__":
    
    '''
    Demo for mnist 10-class classification: 
    '''
    mnist_X_Y = np.load("mnist_feature_label.npz") #load mnist dataset. 
    X = mnist_X_Y["X"]  #X: N x d, where N is the number of examples, and d is the dimension of the feature. 
    Y = mnist_X_Y["Y"]  #Y: N, Y[i] is the label for example X[i]. Y[i] is an integer from [0,1.,..k-2,k-1].
    
    M, W = online_perceptron(X, Y)
    
    
    '''
    do the plot:
        x-axis: t
        y_axis: M
    '''

    #plot code:
    plt.plot(np.arange(1,70001), M)
    plt.xlabel('t')
    plt.ylabel('Mt/t (number of mistakes till step t)')
    plt.show()

    






