##############################################
### author: Abhishek Behera                ###
### email : abhishek.enlightened@gmail.com ###
###         abhishekb@ee.iitb.ac.in        ###
### date  : 29 Mar 2019                    ###
##############################################

import sys,os
import numpy as np
from hmmlearn import hmm

def read_model(model_file,path):
    HMM = open(os.path.join(path,model_file)).readlines()
    HMM = [line.strip() for line in HMM]
    HMM = [line.split('#')[0] for line in HMM]
    HMM = [line for line in HMM if line != '']
    HMM = [line.split(',') for line in HMM]
    return HMM

def BaumWelch_inbuilt(model_file,path=''):
    if path == '':
        path = os.path.realpath( os.path.join(os.getcwd(),os.path.dirname(__file__)))

    HMM = read_model(model_file,path)
    print("Observation_sequence:\n",HMM[2],"\n")
    Obs = [HMM[1].index(o) for o in HMM[2]]
    Obs = np.array(Obs).reshape(-1,1)
    n_states,n_observations = list(map(len,HMM[:2]))
    start_probability = np.array(list(map(float,HMM[3])))
    transition_probability = []
    for i in range(n_states):
        transition_probability += [list(map(float,HMM[4+i]))]
    transition_probability = np.array(transition_probability)
    emission_probability = []
    for a in range(n_observations):
        emission_probability += [list(map(float,HMM[4+n_states+a]))]
    emission_probability = np.array(emission_probability)

    print("Start_probability:\n",np.round(start_probability,3),"\n")
    print("Transition_Probability:\n",np.round(transition_probability,3),"\n")
    print("Emission_Probability:\n",np.round(emission_probability,3),"\n")

    model = hmm.CategoricalHMM(n_components=n_states,n_iter=1000,tol=1e-9,init_params=" ",params="te")
    model.startprob_=start_probability
    model.transmat_=transition_probability
    model.emissionprob_=emission_probability

    model = model.fit(Obs)

    print("==================================\n")
    print("After Training with Baum-Welch algorithm:")

    print("Start probability:\n",np.round(model.startprob_,3),"\n")
    print("Transition probability:\n",np.round(model.transmat_,3),"\n")
    print("Emission probability:\n",np.round(model.emissionprob_,3),"\n")

if __name__ == '__main__':
    for i in range(1,len(sys.argv)):
        BaumWelch_inbuilt(sys.argv[i])
