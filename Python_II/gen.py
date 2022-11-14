from crnsimulator import ReactionGraph
from IPython.display import Image
import numpy as np
from HMMCRN import *
from hmmlearn import hmm
import crnsimulator
import sys
sys.setrecursionlimit(10000000)
states = ["H1","H2"]
n_states = len(states)

observations = ["V1", "V2"]
n_observations = len(observations)

start_probability = np.array([0.5,0.5])

transition_probability=np.array([[0.2, 0.8],
       [0.7, 0.3]])

emission_probability = np.array([[0.75, 0.25],
       [0.3, 0.7]])
model = hmm.MultinomialHMM(n_components=n_states,n_iter=1000,tol=1e-15,init_params=" ",params="te")
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability
np.random.seed(77)

X, Z = model.sample(10000)
R=HMMCRN(['H1','H2'],['V1','V2'])
ob=X
R.Ob=ob
Tr=R.EM()
RG=ReactionGraph(Tr)
print("Writing the File now")
filename,odename=RG.write_ODE_lib(filename='1-10000.py')
