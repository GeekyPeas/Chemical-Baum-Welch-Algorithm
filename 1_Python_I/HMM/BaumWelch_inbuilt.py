##############################################
### author: Abhishek Behera                ###
### email : abhishek.enlightened@gmail.com ###
###         abhishekb@ee.iitb.ac.in        ###
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

def BaumWelch(model_file,path=''):
    if path == '':
        path = os.path.realpath( os.path.join(os.getcwd(),os.path.dirname(__file__)))

    HMM = read_model(model_file,path)
    Obs = [HMM[1].index(o) for o in HMM[2]]
    Obs = np.array(Obs).reshape(-1,1)
    n_states,n_observations = map(len,HMM[:2])
    start_probability = np.array(map(float,HMM[3]))
    transition_probability = []
    for i in range(n_states):
        transition_probability += [map(float,HMM[4+i])]
    transition_probability = np.array(transition_probability)
    emission_probability = []
    for a in range(n_observations):
        emission_probability += [map(float,HMM[4+n_states+a])]
    emission_probability = np.array(emission_probability)

    model = hmm.MultinomialHMM(n_components=n_states,n_iter=500,tol=1e-10,init_params="")
    model.startprob_=start_probability
    model.transmat_=transition_probability
    model.emissionprob_=emission_probability

    print("Viterbi:\n",model.decode(Obs, algorithm="viterbi"),"\n")

    model = model.fit(Obs)
    print("Monitor:\n",model.monitor_,"\n")
    print("Start probability:\n",model.startprob_,"\n")
    print("Transition probability:\n",model.transmat_,"\n")
    print("Emission probability:\n",model.emissionprob_,"\n")

    

if __name__ == '__main__':
    for i in range(1,len(sys.argv)):
        BaumWelch(sys.argv[i])
