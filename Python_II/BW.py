import numpy as np
from hmmlearn import hmm
import csv
import matplotlib.pyplot as plt
import timeit

def BWlib(theta0,theta,psi,X,it,plot=1):
    Obs=X.T.reshape(-1,1)
    n_states=len(theta)
    n_observations=len(psi[0])
    T=len(X)
    n=T
    iteration = [0]
    T0=[theta0]
    TT=[theta]
    T1=[psi]
    model=hmm.CategoricalHMM(n_components=n_states,n_iter=1,tol=1e-9,init_params=" ",params="te")
    model.startprob_=theta0
    model.transmat_=theta
    model.emissionprob_=psi            
    for s in range(1,it):
        model.fit(X)
        iteration.append(s)
        T0.append(model.startprob_)
        TT.append(model.transmat_)
        T1.append(model.emissionprob_)
    model = hmm.CategoricalHMM(n_components=n_states,n_iter=1000,tol=1e-9,init_params=" ",params="te")
    model.startprob_=theta0
    model.transmat_=np.copy(theta)
    model.emissionprob_=np.copy(psi)    
    print(np.round(model.transmat_,3))
    print(np.round(model.emissionprob_,3))
    print(model.score(Obs))    
    if(plot):
        L1=[]
        L2=[]
        for i in range(len(iteration)):
            T=TT[i]
            Psi=T1[i]
            model.transmat_=T
            model.emissionprob_=Psi
            L1.append(model.score(Obs))
            L2.append(model.decode(Obs, algorithm="viterbi")[0])
        plt.plot(iteration,L1)
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood of the observed sequence')
        plt.show()
        plt.plot(iteration,L2)
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood of the decoded hidden state')
        plt.show()
    return L1
def BWlib2(theta0,theta,psi,X,it,plot=1):
    Obs=X.T.reshape(-1,1)
    n_states=len(theta)
    n_observations=len(psi[0])
    T=len(X)
    n=T
    iteration = [0]
    T0=[theta0]
    TT=[theta]
    T1=[psi]
    model=hmm.CategoricalHMM(n_components=n_states,n_iter=1,tol=1e-9,init_params=" ")
    model.startprob_=theta0
    model.transmat_=theta
    model.emissionprob_=psi            
    for s in range(1,it):
        model.fit(X)
        iteration.append(s)
        T0.append(model.startprob_)
        TT.append(model.transmat_)
        T1.append(model.emissionprob_)
    model = hmm.CategoricalHMM(n_components=n_states,n_iter=1000,tol=1e-9,init_params=" ")
    model.startprob_=theta0
    model.transmat_=np.copy(theta)
    model.emissionprob_=np.copy(psi)
    print(np.round(model.startprob_,3))
    print(np.round(model.transmat_,3))
    print(np.round(model.emissionprob_,3))
    print(model.score(Obs))    
    if(plot):
        L1=[]
        L2=[]
        for i in range(len(iteration)):
            model.startprob_=T0[i]
            model.transmat_=TT[i]
            model.emissionprob_=T1[i]
            L1.append(model.score(Obs))
            L2.append(model.decode(Obs, algorithm="viterbi")[0])
        plt.plot(iteration,L1)
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood of the observed sequence')
        plt.show()
        plt.plot(iteration,L2)
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood of the decoded hidden state')
        print(model.score(Obs))    
        print("Final Probabilities")
        print(np.round(model.startprob_,3))
        print(np.round(model.transmat_,3))
        print(np.round(model.emissionprob_,3))
        plt.show()
    return L1    
def init_simulator(n,m,l,I,filename):
    E=[]
    for t in range(1,l+1):
        for i in range(1,n+1):
            E.append('A'+repr(i)+'_'+repr(t))
    for t in range(1,l+1):
        for i in range(1,n+1):
            E.append('B'+repr(i)+'_'+repr(t))
    for t in range(1,l+1):
        for i in range(1,m+1):
            E.append('E'+repr(i)+'_'+repr(t))
    for t in range(1,l+1):
        for i in range(1,n+1):
            E.append('G'+repr(i)+'_'+repr(t))
    for i in range(n+1):
        for j in range(1,n+1):
            E.append('T'+repr(i)+repr(j))
    for i in range(1,n+1):
        for k in range(1,m+1):
            E.append('T'+repr(i)+'_'+repr(k))
    for t in range(1,l):
        for i in range(1,n+1):
            for j in range(1,n+1):
                E.append('Xi'+repr(i)+repr(j)+'_'+repr(t))
    dE=E.copy()
    dE[0]='dA1_1'
    res=[word + 'dt' for word in dE]
    dE=',d'.join(res)
    lines = open(filename).read().splitlines()
    Lstart=1
    for line in lines: 
        if line == "def odesystem(p0, t0, r):":
            break
        Lstart+=1
    lines[Lstart]='    '+','.join(E)+'=p0'
    lines[Lstart+4+len(E)]='    return np.array(['+dE+'])'
    Lstart=1
    for line in lines: 
        if line == "        logger.warning('Deprecated argument: --pyplot_labels.')":
            break
        Lstart+=1
    lines[Lstart+1]="    svars="+str(E)
    lines[Lstart+3] = "    p0 ="+str(I)
    open(filename,'w').write('\n'.join(lines))
    start=E.index('T01')
    return start
def init_simulator2(n,m,l,I,filename):
    E=[]
    for t in range(1,l+1):
        for i in range(1,n+1):
            E.append('A'+repr(i)+'_'+repr(t))
    for t in range(1,l+1):
        for i in range(1,n+1):
            E.append('B'+repr(i)+'_'+repr(t))
    for t in range(1,l+1):
        for i in range(1,m+1):
            E.append('E'+repr(i)+'_'+repr(t))
    for t in range(1,l+1):
        for i in range(1,n+1):
            E.append('G'+repr(i)+'_'+repr(t))
    for i in range(n+1):
        for j in range(1,n+1):
            E.append('T'+repr(i)+repr(j))
    for i in range(1,n+1):
        for k in range(1,m+1):
            E.append('T'+repr(i)+'_'+repr(k))
    for t in range(1,l):
        for i in range(1,n+1):
            for j in range(1,n+1):
                E.append('Xi'+repr(i)+repr(j)+'_'+repr(t))
    dE=E.copy()
    dE[0]='dA1_1'
    res=[word + 'dt' for word in dE]
    dE=',d'.join(res)
    lines = open(filename).read().splitlines()
    Lstart=1
    for line in lines: 
        if line == "def odesystem(t0, p0):":
            break
        Lstart+=1
    lines[Lstart]='    '+','.join(E)+'=p0'
    lines[Lstart+4+len(E)]='    return np.array(['+dE+'])'
    Lstart=1
    for line in lines: 
        if line == "        logger.warning('Deprecated argument: --pyplot_labels.')":
            break
        Lstart+=1
    lines[Lstart+1]="    svars="+str(E)
    lines[Lstart+3] = "    p0 ="+str(I)
    open(filename,'w').write('\n'.join(lines))
    start=E.index('T01')
    return start    
def crn_Liklihood(filenamecsv,start,Obs,n=2,m=2):
    model = hmm.CategoricalHMM(n_components=n,n_iter=1,tol=1e-9,init_params=" ",params="tes")
    t = []
    T0=[]
    TT=[]
    T1=[]
    c=start+1
    with open(filenamecsv,'r') as csvfile:
        #next(csvfile)
        plots = csv.reader(csvfile, delimiter=' ')
        for row in plots:
            t.append(float(row[0]))
            T0.append(np.array(row[c:c+n]).astype(float))
            TT.append(np.array(row[c+n:c+n+n*n]).reshape(n,n).astype(float))
            T1.append(np.array(row[c+n+n*n:c+n+n*n+n*m]).reshape(n,m).astype(float))       
    L1=[]
    L2=[]
    for i in range(len(t)):
        model.startprob_=T0[i]
        model.transmat_=TT[i]
        model.emissionprob_=T1[i]
        L1.append(model.score(Obs))
        L2.append(model.decode(Obs, algorithm="viterbi")[0])
    plt.plot(t,L1)
    plt.xlabel('time')
    plt.ylabel('Log Likelihood of the observed sequence')
    plt.show()
    plt.plot(t,L2)
    plt.xlabel('time')
    plt.ylabel('Log Likelihood of the decoded hidden state')
    plt.show()
    print(model.score(Obs))
    print("CRN_Transition fit:\n",np.round(model.transmat_,3))
    #print("True_Transition:\n",transition_probability)
    print(" ")
    print("CRN_Emission fit:\n",np.round(model.emissionprob_,3))
    #print("True_Emission:\n",emission_probability)
    return t,L1,T0,TT,T1
def print_conc(start,Y1,Y2,n=2,m=2):
    I=np.array(Y1).astype(float)
    spt0=np.array(I[start:start+n])
    spt=np.array(I[start+n:start+n+n*n]).reshape(n,n)
    spe=np.array(I[start+n+n*n:start+n+n*n+n*m]).reshape(n,m)
    print("CRN initialization")
    print("Starting Prob:\n",spt0)
    print("Transition:\n",spt)
    print("Emission:\n",spe)
    I=np.array(Y2).astype(float)
    print("\nCRN Equilibrium")
    pt0=np.round(np.array(I[start:start+n]),3)
    pt=np.round(np.array(I[start+n:start+n+n*n]).reshape(n,n),3)
    pe=np.round(np.array(I[start+n+n*n:start+n+n*n+n*m]).reshape(n,m),3)
    print("Starting Prob:\n",pt0)
    print("Transition:\n",pt)
    print("Emission:\n",pe)