#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:53:02 2018

@author: lucien
"""
from tqdm import tqdm  # 1000*1000 calcul raisonnable    A changer;
# normalisation+init des inputs + descente grad
import numpy as np     #faire plusieurs focntions : descente grad, lambda et dropout
import pickle
import matplotlib.pyplot as plt

dataset_path='/home/lucien/Documents/RT-1/PM_DeepLearning/logisticReg/data_set_1/cifar-10-batches-py/'
m_train =2000# max 49 999
m_test=	300	#max 9 999

num_pix=32
n_inputLayer=num_pix*num_pix*3   #number of elements in input layer

networkArch=[n_inputLayer,300,10]  #number of elements of each layer   
#79% avec input,20,1 ,LR=0.05,2500,lam=0.5

#Classes : 0 avion,1 voiture,2 oiseau,3 chat,4 cerf,5 chien, 6 grenouille
				#7 cheval, 8 bateau, 9 camion
learning_rate=0.006
num_iters=150
lambd=0.05 #80% avec mtrain=800,20 n_h,lR =0.08,numIter=2500,lambd=4

def initDatasets(m_train,m_test):
    
	#ouverture pickle du dictionnaire
    fich=open(dataset_path+"data_batch_"+str(1),'rb')
    dict = pickle.load(fich,encoding='bytes')  #pickle.load(fich, encoding='bytes')
	#extraction des images (matrice 3072*10000)
    X_train=np.array(dict[b'data'].T)
    y_train=np.asarray(dict[b'labels'])  #shape 10 000,
    
    #=========================================
   # for i in range(2,6):
   #     fich=open(dataset_path+"data_batch_"+str(i),'rb')
   #     dict = pickle.load(fich,encoding='bytes')  #pickle.load(fich, encoding='bytes')
   # 	#extraction des images (matrice 3072*10000)
   #     X_train=np.concatenate((X_train , np.asarray(dict[b'data'].T) ),axis = 1)
   #     y_train=np.concatenate((y_train, np.asarray(dict[b'labels'])))
   # Charge tous les autres batch bouffe de la ram
    #======================================
    
    fich=open(dataset_path+"test_batch",'rb')
    dict = pickle.load(fich,encoding='bytes')
    X_test= np.asarray(dict[b'data'].T)
    y_test= np.asarray(dict[b'labels'])
    
    X_train = X_train[:,0:m_train]
    y_train = y_train[0:m_train]
    X_test = X_test[:,0:m_test]
    y_test = y_test[0:m_test]
    
    #mise des Y sous forme vectorielle : passe de y=4 pour classe 5 à y=[0 0 0 0 1 0 0 0 0]
    y_test_M = np.zeros((m_test,10))
    y_train_M = np.zeros((m_train,10))
    
    for i in range(0,m_test):
        y_test_M[i,y_test[i]] = 1
    for i in range(0,m_train):
        y_train_M[i,y_train[i]] = 1
    
    return X_train,y_train_M,X_test,y_test_M

def plotImage(X,idx,num_pix):

	rgb=X[:,idx]    
	img=rgb.reshape(3,num_pix,num_pix).transpose([1,2,0])
	plt.imshow(img)
	plt.show()

def standardizeDataset(X_train,X_test):
    #normalisation simplifiée	
    return (X_train - np.mean(X_train))/np.std(X_train) , (X_test - np.mean(X_test))/np.std(X_test)

def initialiseParameters(networkArch):

    parameters = dict()
    L = len(networkArch)
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(networkArch[l],networkArch[l-1])*np.sqrt(2/networkArch[l-1])
        parameters["b"+str(l)] = np.zeros((networkArch[l],1))    
    return parameters

def ReLu(z):
    z[np.where(z <= 0)] = 0
    return z

def sigmoid(z):
	s= 1 / (1 + np.exp( -z))
	return s

def derivativeRelu(z):
    z[np.where(z <= 0)] = 0
    z[np.where(z > 0)] = 1
    return z

def derivativeSigmoid(z):
    z = sigmoid(z)*(1 - sigmoid(z) )
    return z
    
def forwardPropagation(parameters,X):
    
    cache=dict()
    L = len(parameters)//2 #np de layers
    cache["A"+str(0)] = X
    A=X
   
    for l in range(1,L):
        Z = np.dot(parameters["W"+str(l)],A) + parameters["b"+str(l)]
        cache["Z"+str(l)] = Z
        A=ReLu(Z)
        cache["A"+str(l)] = A
        
    Z_L = np.dot(parameters["W"+str(L)] , A) + parameters["b"+str(L)]
    cache["Z"+str(L)] = Z_L
    A_L = sigmoid(Z_L)
    cache["A"+str(L)] = Z_L
    
    return A_L , cache

def computeCost(A_L , y_train, parameters,lambd ):
    
    L = len(parameters)//2
    regularisation = 0
    m=np.shape(y_train)[0]
    for l in range(1,L+1):
        regularisation = regularisation + np.sum( np.square(parameters["W"+str(l)]) )
    
    J = (1/m) * np.sum(  -np.multiply(y_train.T, np.log(A_L)) - np.multiply((1-y_train).T , np.log(1-A_L))  ) 
    
    J = J + regularisation*lambd/(2*m)

    return J

def derivativeCostFunction(A_L,y):
    dAL=- (np.divide(y.T, A_L) - np.divide(1 - y.T, 1 - A_L)) 
    return dAL

def backPropagation(parameters, y_train, A_L, cache,lambd):
    L = len(parameters)//2
    m = np.shape(y_train)[0]
    grads=dict()

    dAL = derivativeCostFunction(A_L ,y_train)
    dZL = dAL*derivativeSigmoid(cache["Z"+str(L)])
  
    grads["dW"+str(L)] = (1/m)* np.dot( dZL , cache["A"+str(L-1)].T )+(lambd/m)*parameters["W"+str(L)]
    grads["db"+str(L)] = (1/m)* np.sum( dZL, axis=1, keepdims=True)
    dAl_1 = np.dot( parameters["W"+str(L)].T , dZL)
    
    for l in reversed(range(1,L)):
        
        dZl = dAl_1 * derivativeRelu(cache["Z"+str(l)])
        grads["dW"+str(l)] =(1/m)* np.dot( dZl,cache["A"+str(l-1)].T )+(lambd/m)*parameters["W"+str(l)]
        grads["db"+str(l)] =(1/m)* np.sum( dZl, axis=1, keepdims=True)
        dAl_1 = np.dot( parameters["W"+str(l)].T , dZl )

    return grads

def updateParameters(parameters,grads,Learning_rate):
    L = len(parameters)//2

    for l in range(1,L+1):
        
        parameters["W"+str(l)] = parameters["W"+str(l)] - Learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - Learning_rate*grads["db"+str(l)]
    return parameters

def plotCost(j_history,num_iters,learning_rate,j_test_history):
    x=np.arange(0,num_iters)
    y=j_history
    y2=j_test_history
    plt.plot(x,y,'b',linewidth=2.0,label="train error")
    plt.plot(x,y2,'r',linewidth=2.0,label="test error")
    plt.legend()
    plt.title("learning_rate =%f" %(learning_rate))
    plt.show()

def predict(X,y,parameters):
    
    A_L,cache = forwardPropagation(parameters,X) #A4 : 1,200 et y: 200,1
    A_L=A_L.T
    max_indices_pred= A_L.argmax(axis=1)
    max_indices_test = y.argmax(axis=1)
 
    accuracy=np.mean(max_indices_pred == max_indices_test)
    return accuracy

def gradientChecking(grads,parameters,y_train,X_train,lambd):
    #convertir dictionnaire grads en vecteur W1,b1,W2,b2
    L = len(parameters)//2
    epsilon = 1e-7
    
    vect_grads=grads["dW1"].flatten()
    vect_grads = np.concatenate((vect_grads,grads["db1"].flatten()),axis=0)

    for i in range(2,L+1):
        vect_grads= np.concatenate((vect_grads,grads["dW"+str(i)].flatten()),axis=0)
        vect_grads = np.concatenate((vect_grads,grads["db"+str(i)].flatten()),axis=0)

    #calcul approx gradient
    
    grad_approx=np.zeros((np.shape(vect_grads)[0]))

    #ajouter un thet+   / et un theta moins dans deux vecteurs
    # remetre en dictionnaire de matrices
    #calculer le cout
    #calculer l'écart
    rang=0
    for l in range(1,L+1):

        for i in range(0,np.shape(parameters["W"+str(l)])[0]):
            for j in range(0,np.shape(parameters["W"+str(l)])[1]):
                
                parameters["W"+str(l)][i,j] = parameters["W"+str(l)][i,j] + epsilon  # +e
                A_L,cache = forwardPropagation(parameters,X_train)
                J1=computeCost(A_L,y_train,parameters,lambd)
                
                parameters["W"+str(l)][i,j] = parameters["W"+str(l)][i,j] - 2*epsilon # -e
                A_L,cache = forwardPropagation(parameters,X_train)
                J2=computeCost(A_L,y_train,parameters,lambd)
                
                grad_approx[rang] = (J1 - J2 )/ (2*epsilon)
                rang=rang+1
                parameters["W"+str(l)][i,j] = parameters["W"+str(l)][i,j] + epsilon  # retour au début     
         #calcul pour le bias
        print("i ="+str(i))
        parameters["b"+str(l)] = parameters["b"+str(l)] + epsilon
        A_L,cache = forwardPropagation(parameters,X_train)
        J1=computeCost(A_L,y_train,parameters,lambd)
         
        parameters["b"+str(l)] = parameters["b"+str(l)] - 2*epsilon # -e
        A_L,cache = forwardPropagation(parameters,X_train)
        J2=computeCost(A_L,y_train,parameters,lambd)
                                             
        grad_approx[rang] = (J1 - J2 )/ (2*epsilon)
        rang=rang+1
        parameters["b"+str(l)] = parameters["b"+str(l)]+ epsilon  # retour au début
    #gradient checking
    error = np.linalg.norm( np.abs(grad_approx - vect_grads)) / (np.linalg.norm(grad_approx) + np.linalg.norm(vect_grads))
    print("erreur sur le grad :"+str(error))
    if error >= 1e-3 :
        print("erreur non acceptable")                                  
    else:
        print("bon gradient")
        
#-------------------------  Initialisation    ----------------------------
        
X_train, y_train, X_test, y_test = initDatasets(m_train,m_test)
X_train , X_test = standardizeDataset(X_train,X_test)
X_train = X_train[:,0:m_train]
y_train = y_train[0:m_train,:]
X_test = X_test[:,0:m_test]
y_test = y_test[0:m_test,:]

print("Deep neural network 1 \n\nBatch gradient descent, gradient checking, L2 Regularisation")

plotImage(X_train,0,32)

parameters = initialiseParameters(networkArch)
j_history=np.zeros((num_iters,1))
j_test_history=np.zeros((num_iters,1))

J=0
iteration=0

pbar = tqdm(range(0,num_iters))

for iter in pbar:         
    A_L , cache = forwardPropagation(parameters,X_train)
    J = computeCost(A_L,y_train,parameters,lambd)
    
    
    grads = backPropagation(parameters,y_train,A_L,cache,lambd) 
    #gradientChecking(grads,parameters,y_train,X_train,lambd)
    parameters=updateParameters(parameters,grads,learning_rate)
    
    pbar.set_description("cost iter %f"  %J)
    
    j_history[iter,0] = J
    A_L_test, cache2 = forwardPropagation(parameters,X_test)
    j_test = computeCost(A_L_test,y_test,parameters,lambd)
    j_test_history[iter,0] = j_test
    
accuracy_train=predict(X_train,y_train,parameters)
accuracy_test=predict(X_test,y_test,parameters)

print("alpha=%f, trainSet:%f testSet:%f" %(learning_rate,accuracy_train, accuracy_test))
plotCost(j_history,num_iters,learning_rate,j_test_history)

