##############################################
### author: Abhishek Behera                ###
### email : abhishek.enlightened@gmail.com ###
###         abhishekb@ee.iitb.ac.in        ###
### date  : 24 May 2018                    ###
##############################################

import numpy as np
from scipy.integrate import odeint
from scipy.special import comb,factorial
import matplotlib.pyplot as plt
import sys

if sys.version_info[0]==2:
    range = xrange

def odes(conc,time,reactants,products,rates):
    return (products - reactants).dot(np.prod(conc**reactants.T,axis=1)*rates)

def DMAK_CPU(reactants,products,rates,initial_conc,duration,num_steps,keep_path=False,do_plot=False,species=None):
    reactants = np.array(reactants)
    products = np.array(products)
    rates = np.array(rates)*1.0
    if species is None:
        species =['S_'+str(i+1) for i in range(reactants.shape[0])]
        
    t_index = np.linspace(0, duration, num_steps)
    conc = initial_conc
    output = odeint(odes, conc, t_index, args = (reactants,products,rates))
    if keep_path:
        if do_plot:
            for i in range(output.shape[1]):
                label = species[i]
                plt.plot(t_index, output[:, i], label = '$'+label+'$')
            plt.legend(loc='best')
            plt.xlabel('Time')
            plt.ylabel('Concentration')
            plt.grid()
            plt.show()
        return output
    else:
        return output[-1]
