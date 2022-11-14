##############################################
### author: Abhishek Behera                ###
### email : abhishek.enlightened@gmail.com ###
###         abhishekb@ee.iitb.ac.in        ###
### date  : 29 Mar 2019                    ###
##############################################

import sys,os
import numpy as np

def read_model(model_file,path):
    HMM = open(os.path.join(path,model_file)).readlines()
    HMM = [line.strip() for line in HMM]
    HMM = [line.split('#')[0] for line in HMM]
    HMM = [line for line in HMM if line != '']
    HMM = [line.split(',') for line in HMM]
    return HMM

def BaumWelch(model_file,path=''):
    if path == '':
        path = os.path.realpath( os.path.join(os.getcwd(),os.path.dirname(__file__)))

    HMM = read_model(model_file,path)
    Obs = [HMM[1].index(o) for o in HMM[2]]
    T = len(Obs)
    X = np.array(Obs)
    n_states,n_observations = map(len,HMM[:2])
    start_probability = np.array(HMM[3]).astype(float)
    transition_probability = []
    for i in range(n_states):
        transition_probability += [np.array(HMM[4+i]).astype(float)]
    transition_probability = np.array(transition_probability)
    emission_probability = []
    for a in range(n_observations):
        emission_probability += [np.array(HMM[4+n_states+a]).astype(float)]
    emission_probability = np.array(emission_probability)

    print("Observation_sequence:\n",X,"\n")
    print("Start_probability:\n",start_probability,"\n")
    print("Transition_Probability:\n",transition_probability,"\n")
    print("Emission_Probability:\n",emission_probability,"\n")

    A=np.zeros([T,n_states])
    B=np.zeros([T,n_states])
    for j in range(n_states):
        B[-1,j]=1.0            
    Xi=np.zeros([T-1,n_states,n_states])
    Psi=np.zeros([n_states])
    G=np.zeros([T,n_states])
    n=T
    
    theta0 = start_probability
    theta = transition_probability
    psi = emission_probability

    for s in range(200):    
        for j in range(n_states):
            A[0,j]=theta0[j]*psi[j,X[0]]
        for t in range(1,n):
            for j in range(n_states):
                A[t,j]=0
                for i in range(n_states):
                    A[t,j]=A[t,j]+A[t-1,i]*theta[i,j]*psi[j,X[t]]
        for t in range(T-2,-1,-1):
            for i in range(n_states):
                B[t,i]=0
                for j in range(n_states):
                    B[t,i]=B[t,i]+theta[i,j]*psi[j,X[t+1]]*B[t+1,j]
        for t in range(n-1):
            for i in range(n_states):
                for j in range(n_states):
                    S=0.0
                    for m in range(n_states):
                        S=S+A[t,m]*B[t,m]
                    Xi[t,i,j]=(A[t,i]*theta[i,j]*psi[j,X[t+1]]*B[t+1,j])/float(S)
        for t in range(n):
            for i in range(n_states):
                S=0.0
                for j in range(n_states):
                    S=S+A[t,j]*B[t,j]
                G[t,i]=A[t,i]*B[t,i]/S
        for i in range(n_states):
            for j in range(n_states): 
                theta[i,j]=sum(Xi[:,i,j])/sum(sum(Xi[:,i,:]))
            for k in range(n_observations):
                psi[i,k]=0
                for t in range(n):
                    if k==X[t]:
                        psi[i,k]=psi[i,k]+G[t,i]
                psi[i,k]=psi[i,k]/sum(G[:,i]) 

    print("==================================\n")
    print("After Training:")
    print("Alpha:\n",A,"\n")
    print("Beta:\n",B,"\n")
    print("Gamma:\n",G,"\n")
    print("Xi:\n",Xi,"\n")
    print("Transition_Probability:\n",theta,"\n")
    print("Emission_Probability:\n",psi,"\n")

if __name__ == '__main__':
    for i in range(1,len(sys.argv)):
        BaumWelch(sys.argv[i])
