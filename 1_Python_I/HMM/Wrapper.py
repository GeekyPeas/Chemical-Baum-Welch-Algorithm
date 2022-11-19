##############################################
### author: Abhishek Behera                ###
### email : abhishek.enlightened@gmail.com ###
###         abhishekb@ee.iitb.ac.in        ###
##############################################

import os,sys
import numpy as np
from Model_to_CRN import translate_model as translate
sys.path.insert(0, os.path.join('..','..','CRN_Engines'))
from DMAK_CPU import DMAK_CPU as DMAK

for i in range(1,len(sys.argv)):
    print("\n"+sys.argv[i])
    reaction_system = translate(sys.argv[i],False,False)
    species = np.array(reaction_system[0])
    reactants = np.array(reaction_system[1][0])
    products = np.array(reaction_system[1][1])
    rates = np.array(reaction_system[1][2])
    initial = np.array(reaction_system[2])    
    
    duration = 2000
    num_step = 1000

    print("species:",species)
    print("reactants:",reactants)
    print("products:",products)
    print("rates:",rates)
    print("initial:",initial)

    output = DMAK(reactants,products,rates,initial,duration,num_step,True,True,species)[-1]
    
    for i in range(len(species)):
        print(species[i],output[i])

    print("\n=================================")
    derivatives = (products - reactants).dot(np.prod(output**reactants.T,axis=1)*rates)
    print(derivatives)
