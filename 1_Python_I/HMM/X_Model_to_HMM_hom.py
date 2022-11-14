##############################################
### author: Abhishek Behera                ###
### email : abhishek.enlightened@gmail.com ###
###         abhishekb@ee.iitb.ac.in        ###
### date  : 19 Mar 2019                    ###
##############################################

import numpy as np
import sys, os
import itertools

sys.path.insert(0, os.path.join('..','..','CRN_Engines'))
from Reaction_Utils import display, write

if sys.version_info[0] == 2:
    range = xrange

def read_model(model_file,path):
    HMM = open(os.path.join(path,model_file)).readlines()
    HMM = [line.strip() for line in HMM]
    HMM = [line.split('#')[0] for line in HMM]
    HMM = [line for line in HMM if line != '']
    HMM = [line.split(',') for line in HMM]
    return HMM

def sp(name,index):
    if name not in ('\\xi','\\gamma'):
        specie = name+'_{'+','.join(map(str,index))+'}'
    else:
        specie = name+'_{'+str(index[0])+'}'+'('+','.join(map(str,index[1:]))+')'
    return specie

def make_species(meta_species):
    species = []
    for meta in meta_species:
        name = meta[0]
        indices = [range(1,n+1) if isinstance(n,int)\
                   else n for n in meta[1]]
        for index in itertools.product(*indices):
            species += [sp(name,index)]
    return species

def cx(meta_cplx,loops,var,obs,species):
    # print("meta_cplx:",meta_cplx)
    # print("loops:",loops)
    # print("var:",var)
    # print("obs:",obs)
    # print("species:",species)
    arr = len(meta_cplx)
    # print("arr:",arr)
    indices = [range(r[1][0],r[1][1]+1) for r in loops]
    # print("indices:",indices)
    dummies = [r[0] for r in loops]
    # print("dummies:",dummies)
    rid = dummies.index(var[0])
    # print("rid:",rid)
    complexes = []
    for index in itertools.product(*indices):
        # print("index:",index)
        cplx = meta_cplx
        # print("cplx:",cplx)
        cplx = list(map(lambda c: c.replace('<sub>',obs[index[rid]-1]), cplx))
        # print("cplx:",cplx)
        cplx = list(map(lambda c: c.replace('<inc>',str(index[rid]+1)), cplx))
        # print("cplx:",cplx)
        cplx = list(map(lambda c: c.replace('<dec>',str(index[rid]-1)), cplx))
        # print("cplx:",cplx)
        for i in range(len(loops)):
            cplx = list(map(lambda c: c.replace(dummies[i],str(index[i])), cplx))
            # print("cplx:",cplx)
        cplx = list(map(species.index,cplx))
        # print("cplx:",cplx)
        rep = np.zeros(len(species))
        # print("rep:",rep)
        np.put(rep,cplx,np.ones(arr))
        complexes += [rep]
    return np.array(complexes)
                                  
def make_reactions(meta_reactions,obs,species):
    reactants = []
    products = []
    for meta in meta_reactions:
        loops,var=meta[-2:]
        reactant,product,f_catalyst,b_catalyst=map(lambda c: cx(c,loops,var,obs,species),meta[:-2])
        #forward reaction:
        f_reactant = reactant+f_catalyst
        reactants += f_reactant.tolist()
        f_product = product+f_catalyst
        products += f_product.tolist()
        #backward reaction:
        b_reactant = product+b_catalyst
        reactants += b_reactant.tolist()
        b_product = reactant+b_catalyst
        products += b_product.tolist()
    rates = np.ones(len(reactants))        
    reactants = np.array(reactants,dtype=int).transpose()
    products = np.array(products,dtype=int).transpose()
    return [reactants,products,rates]


def translate_model(model_file,
                    display_reaction=False,
                    save_reaction=False,path=''):
    if path == '':
        path = os.path.realpath( os.path.join(os.getcwd(),os.path.dirname(__file__)))
    hmm = read_model(model_file,path)
    obs = hmm[2]
    N,M,L = map(len,hmm[:3])
    values = [float(item) for sublist in hmm[3:] for item in sublist]

    a,b,g,x,t,p,o = ['\\alpha','\\beta','\\gamma','\\xi','\\theta','\\psi','O']
    meta_species = [[a,[L,N]],[b,[L,N]],[g,[L,N]],[x,[L,N,N]],[t,['*',N]],[t,[N,N]],[p,[N,M]],[o,[L,M]]]
    species = make_species(meta_species)

    # for s in range(len(species)):
    #     print species[i],s
    #TODO: If I can code up for substituting arbitrary functions of arbitrary variables
    #I have a pretty good translator for arbitrary CRN schemes actually

    meta_reactions = []
    # Forward Reaction Network:
    meta_reactions += [[[sp(a,['1','<j>'])],
                        [sp(a,['1',str(N)])],
                        [sp(t,['*',str(N)]),sp(p,[str(N),obs[0]])],
                        [sp(t,['*','<j>']),sp(p,['<j>',obs[0]])],
                        [('<j>',[1,N-1]),('<l>',[1,1])],
                        ['<l>']]]
    meta_reactions += [[[sp(a,['<dec>','<i>']),sp(a,['<l>','<j>'])],
                        [sp(a,['<dec>','<i>']),sp(a,['<l>',str(N)])],
                        [sp(t,['<i>',str(N)]),sp(p,[str(N),'<sub>'])],
                        [sp(t,['<i>','<j>']),sp(p,['<j>','<sub>'])],
                        [('<i>',[1,N]),('<j>',[1,N-1]),('<l>',[2,L])],
                        ['<l>']]]
    # Backward Reaction Network:
    meta_reactions += [[[sp(b,['<l>','<i>']),sp(b,['<dec>','<j>'])],
                        [sp(b,['<l>','<i>']),sp(b,['<dec>',str(N)])],
                        [sp(t,[str(N),'<i>']),sp(p,['<i>','<sub>'])],
                        [sp(t,['<j>','<i>']),sp(p,['<i>','<sub>'])],
                        [('<i>',[1,N]),('<j>',[1,N-1]),('<l>',[2,L])],
                        ['<l>']]]
    # Expectation Step:
    meta_reactions += [[[sp(g,['<l>','<i>'])],
                        [sp(g,['<l>',str(N)])],
                        [sp(a,['<l>',str(N)]),sp(b,['<l>',str(N)])],
                        [sp(a,['<l>','<i>']),sp(b,['<l>','<i>'])],
                        [('<i>',[1,N-1]),('<l>',[1,L])],
                        ['<l>']]]
    meta_reactions += [[[sp(x,['<dec>','<i>','<j>'])],
                        [sp(x,['<dec>',str(N),str(N)])],
                        [sp(a,['<dec>',str(N)]),sp(b,['<l>',str(N)]),sp(t,[str(N),str(N)]),sp(p,[str(N),'<sub>'])],
                        [sp(a,['<dec>','<i>']),sp(b,['<l>','<j>']),sp(t,['<i>','<j>']),sp(p,['<j>','<sub>'])],
                        [('<i>',[1,N]),('<j>',[1,N]),('<l>',[2,L])],
                        ['<l>']]] #!!!!TODO: implement (i,j) != (N,N)
    # Maximization Step:
    meta_reactions += [[[sp(t,['<i>','<j>'])],
                        [sp(t,['<i>',str(N)])],
                        [sp(x,['<l>','<i>',str(N)])],
                        [sp(x,['<l>','<i>','<j>'])],
                        [('<i>',[1,N]),('<j>',[1,N-1]),('<l>',[1,L-1])],
                        ['<l>']]]
    meta_reactions += [[[sp(p,['<j>','<a>'])],
                        [sp(p,['<j>',str(M)])],
                        [sp(g,['<l>','<j>']),sp(o,['<l>',str(M)])],
                        [sp(g,['<l>','<j>']),sp(o,['<l>','<a>'])],
                        [('<a>',[1,M-1]),('<j>',[1,N]),('<l>',[1,L])],
                        ['<l>']]]
    
    hmm_crn = make_reactions(meta_reactions,obs,species)

    #initialization...
    initial = np.zeros(len(species))

    # # Delta initialization:
    # for l in range(1,L+1):
    #     for s in [a,b,g]:
    #         loc = species.index(sp(s,[str(l),str(N)]))
    #         initial[loc] = 1.0
    #     loc = species.index(sp(x,[str(l),str(N),str(N)]))
    #     initial[loc]=1.0

    # Symmetric initialization:
    for l in range(1,L+1):
        for i in range(1,N+1):
            for s in [a,b,g]:
                loc = species.index(sp(s,[str(l),str(i)]))
                initial[loc] = 0.5
            for j in range(1,N+1):
                loc = species.index(sp(x,[str(l),str(i),str(j)]))
                initial[loc] = 0.25

    for i in range(1,N+1):
        loc = species.index(sp(b,[str(L),str(i)]))
        initial[loc] = 1.0

                
    loc_s = species.index(sp(t,['*',str(1)]))
    loc_f = species.index(sp(p,[str(N),str(M)]))
    np.add.at(initial,range(loc_s,loc_f+1),values)
    for l in range(len(obs)):
        loc = species.index(sp(o,[str(l+1),obs[l]]))
        initial[loc] = 1.0
    
    reaction_system = [species,hmm_crn,initial]
    
    if display_reaction:
        display(reaction_system)
    
    if save_reaction:
        output = os.path.join(path,'_CRN.'.join(model_file.split('.')))
        write(reaction_system,output,os.path.basename(__file__))
    return reaction_system

if __name__ == '__main__':
    for i in range(1,len(sys.argv)):
        translate_model(sys.argv[i],True,True)
