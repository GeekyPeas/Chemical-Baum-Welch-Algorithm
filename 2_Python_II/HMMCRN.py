import numpy as np

class HMMCRN:
    def __init__(self, H,S,T=[]):
        self.H = H
        self.S = S
        self.T = T
        self.ECRN=[]
        self.MCRN=[]
        self.EMCRN=[]
        self.Ob=[]
    
    def Obs(self,D):
        for i in D:
            self.Ob.append(self.S.index(i)+1)
    
    def EProj(self):
        N=len(self.H)
        Nn=len(self.S)
        n=len(self.Ob)
        #Forward
        for i in range(1,N):
            for k in range(1,Nn+1):
                self.ECRN.append([['T0'+repr(N),'T'+repr(N)+'_'+repr(k),'E'+repr(k)+'_1','A'+repr(i)+'_'+'1'],['T0'+repr(N),'T'+repr(N)+'_'+repr(k),'E'+repr(k)+'_1','A'+repr(N)+'_'+'1'],1])  
                self.ECRN.append([['T0'+repr(i),'T'+repr(i)+'_'+repr(k),'E'+repr(k)+'_1','A'+repr(N)+'_'+'1'],['T0'+repr(i),'T'+repr(i)+'_'+repr(k),'E'+repr(k)+'_1','A'+repr(i)+'_'+'1'],1])
        for t in range(1,n):
            for i in range(1,N+1):
                for j in range(1,N):
                    for k in range(1,Nn+1):
                        self.ECRN.append([['A'+repr(i)+'_'+repr(t),'T'+repr(i)+repr(N),'T'+repr(N)+'_'+repr(k),'E'+repr(k)+'_'+repr(t+1),'A'+repr(j)+'_'+repr(t+1)],['A'+repr(i)+'_'+repr(t),'T'+repr(i)+repr(N),'T'+repr(N)+'_'+repr(k),'E'+repr(k)+'_'+repr(t+1),'A'+repr(N)+'_'+repr(t+1)],1])  
                        self.ECRN.append([['A'+repr(i)+'_'+repr(t),'T'+repr(i)+repr(j),'T'+repr(j)+'_'+repr(k),'E'+repr(k)+'_'+repr(t+1),'A'+repr(N)+'_'+repr(t+1)],['A'+repr(i)+'_'+repr(t),'T'+repr(i)+repr(j),'T'+repr(j)+'_'+repr(k),'E'+repr(k)+'_'+repr(t+1),'A'+repr(j)+'_'+repr(t+1)],1])
        #Backward
        for t in range(n,1,-1):
            for i in range(1,N):
                for j in range(1,N+1):
                    for k in range(1,Nn+1):
                        self.ECRN.append([['T'+repr(N)+repr(j),'T'+repr(j)+'_'+repr(k),'B'+repr(j)+'_'+repr(t),'E'+repr(k)+'_'+repr(t),'B'+repr(i)+'_'+repr(t-1)],['T'+repr(N)+repr(j),'T'+repr(j)+'_'+repr(k),'B'+repr(j)+'_'+repr(t),'E'+repr(k)+'_'+repr(t),'B'+repr(N)+'_'+repr(t-1)],1])  
                        self.ECRN.append([['T'+repr(i)+repr(j),'T'+repr(j)+'_'+repr(k),'B'+repr(j)+'_'+repr(t),'E'+repr(k)+'_'+repr(t),'B'+repr(N)+'_'+repr(t-1)],['T'+repr(i)+repr(j),'T'+repr(j)+'_'+repr(k),'B'+repr(j)+'_'+repr(t),'E'+repr(k)+'_'+repr(t),'B'+repr(i)+'_'+repr(t-1)],1])
        
        for t in range(1,n):
            for i in range(1,N+1):
                for j in range(1,N+1):
                    for k in range(1,Nn+1):
                        if i!=N or j!=N:
                            self.ECRN.append([['T'+repr(N)+repr(N),'T'+repr(N)+'_'+repr(k),'A'+repr(N)+'_'+repr(t),'B'+repr(N)+'_'+repr(t+1),'E'+repr(k)+'_'+repr(t+1),'Xi'+repr(i)+repr(j)+'_'+repr(t)],['T'+repr(N)+repr(N),'T'+repr(N)+'_'+repr(k),'A'+repr(N)+'_'+repr(t),'B'+repr(N)+'_'+repr(t+1),'E'+repr(k)+'_'+repr(t+1),'Xi'+repr(N)+repr(N)+'_'+repr(t)],1])  
                            self.ECRN.append([['T'+repr(i)+repr(j),'T'+repr(j)+'_'+repr(k),'A'+repr(i)+'_'+repr(t),'B'+repr(j)+'_'+repr(t+1),'E'+repr(k)+'_'+repr(t+1),'Xi'+repr(N)+repr(N)+'_'+repr(t)],['T'+repr(i)+repr(j),'T'+repr(j)+'_'+repr(k),'A'+repr(i)+'_'+repr(t),'B'+repr(j)+'_'+repr(t+1),'E'+repr(k)+'_'+repr(t+1),'Xi'+repr(i)+repr(j)+'_'+repr(t)],1]) 
                                
        for t in range(1,n+1):
            for i in range(1,N):
                        self.ECRN.append([['A'+repr(N)+'_'+repr(t),'B'+repr(N)+'_'+repr(t),'G'+repr(i)+'_'+repr(t)],['A'+repr(N)+'_'+repr(t),'B'+repr(N)+'_'+repr(t),'G'+repr(N)+'_'+repr(t)],1])  
                        self.ECRN.append([['A'+repr(i)+'_'+repr(t),'B'+repr(i)+'_'+repr(t),'G'+repr(N)+'_'+repr(t)],['A'+repr(i)+'_'+repr(t),'B'+repr(i)+'_'+repr(t),'G'+repr(i)+'_'+repr(t)],1])                    
        return(self.ECRN)
    
    def MProj(self):
        N=len(self.H)
        Nn=len(self.S)
        n=len(self.Ob)
        #for i in range(1,N+1):
        #        for j in range(1,N+1):
        #            for t in range(1,n):
        #                self.MCRN.append([['Xi'+repr(t)+'_'+repr(i)+repr(j)],['Xi'+repr(t)+'_'+repr(i)+repr(j),'T'+repr(i)+repr(j)],1])
        #                for k in range(1,N+1):
        #                    self.MCRN.append([['T'+repr(i)+repr(j),'Xi'+repr(t)+'_'+repr(i)+repr(k)],['Xi'+repr(t)+'_'+repr(i)+repr(k)],1])
        for i in range(1,N+1):
                for k in range(1,Nn+1):
                    for t in range(1,n+1):
                        self.MCRN.append([['G'+repr(i)+'_'+repr(t),'E'+repr(k)+'_'+repr(t)],['G'+repr(i)+'_'+repr(t),'E'+repr(k)+'_'+repr(t),'T'+repr(i)+'_'+repr(k)],1])
                        self.MCRN.append([['T'+repr(i)+'_'+repr(k),'G'+repr(i)+'_'+repr(t)],['G'+repr(i)+'_'+repr(t)],1])
        for i in range(1,N+1):
                for j in range(1,N):
                    for t in range(1,n):
                            self.MCRN.append([['Xi'+repr(i)+repr(N)+'_'+repr(t),'T'+repr(i)+repr(j)],['Xi'+repr(i)+repr(N)+'_'+repr(t),'T'+repr(i)+repr(N)],1])
                            self.MCRN.append([['Xi'+repr(i)+repr(j)+'_'+repr(t),'T'+repr(i)+repr(N)],['Xi'+repr(i)+repr(j)+'_'+repr(t),'T'+repr(i)+repr(j)],1])
        #for j in range(1,N+1):
        #        for k in range(1,Nn):
        #           for t in range(1,n+1):
        #                   self.MCRN.append([['G'+repr(t)+'_'+repr(j),'E'+repr(t)+'_'+`Nn`,'T'+repr(j)+'_'+repr(k)],['G'+repr(t)+'_'+repr(j),'E'+repr(t)+'_'+`Nn`,'T'+repr(j)+'_'+`Nn`],1])
        #                   self.MCRN.append([['G'+repr(t)+'_'+repr(j),'E'+repr(t)+'_'+repr(k),'T'+repr(j)+'_'+`Nn`],['G'+repr(t)+'_'+repr(j),'E'+repr(t)+'_'+repr(k),'T'+repr(j)+'_'+repr(k)],1])
        return(self.MCRN)
    def EM(self):
        self.ECRN=[]
        self.MCRN=[]
        self.EMCRN=[]
        self.EMCRN=self.EProj()+self.MProj()
        return(self.EMCRN)
def uni_init(n,m,Obs):
    l=len(Obs)
    N=1.0/n
    M=1.0/m
    xi=1.0/(n*n)
    theta=1.0/(n)
    psi=1.0/(m)
    A=[N]*(l*n)
    B=[N]*((l-1)*n)
    B=B+[1.0]*n
    E=[0.0]*(l*m)
    for i in range(l):
        for k in range(m):
            if Obs[i]==k:
                E[m*i+k]=1.0
    G=[N]*(l*n)
    T=[N]*n
    for i in range(n):
        T=T+[N]*n
    for i in range(n):
        T=T+[M]*m
    Xi=[xi]*((l-1)*n*n)
    X=A+B+E+G+T+Xi
    return X
def ran_init(n,m,Obs,seed=1):
    np.random.seed(seed)
    l=len(Obs)
    N=1.0/n
    M=1.0/m
    xi=1.0/(n*n)
    theta=1.0/(n)
    psi=1.0/(m)
    A=[]
    for i in range(l):
        A=A+np.random.dirichlet(np.ones(n),size=1).tolist()[0]
    B=[]
    for i in range(l-1):
        B=B+np.random.dirichlet(np.ones(n),size=1).tolist()[0]
    B=B+[1.0]*n
    E=[0.0]*(l*m)
    for i in range(l):
        for k in range(m):
            if Obs[i]==k:
                E[m*i+k]=1.0
    G=[]
    for i in range(l):
        G=G+np.random.dirichlet(np.ones(n),size=1).tolist()[0]
    T=[N]*n
    for i in range(n):
        T=T+np.random.dirichlet(np.ones(n),size=1).tolist()[0]
    for i in range(n):
        T=T+np.random.dirichlet(np.ones(m),size=1).tolist()[0]
    Xi=[]
    for t in range(l-1):
            Xi=Xi+np.random.dirichlet(np.ones(n*n),size=1).tolist()[0]
    X=A+B+E+G+T+Xi
    return X

def ran_init2(n,m,Obs,theta0,rA,rB,seed=1):
    np.random.seed(seed)
    l=len(Obs)
    N=1.0/n
    M=1.0/m
    xi=1.0/(n*n)
    theta=1.0/(n)
    psi=1.0/(m)
    A=[]
    for i in range(l):
        A=A+np.random.dirichlet(np.ones(n),size=1).tolist()[0]
    B=[]
    for i in range(l-1):
        B=B+np.random.dirichlet(np.ones(n),size=1).tolist()[0]
    B=B+[1.0]*n
    E=[0.0]*(l*m)
    for i in range(l):
        for k in range(m):
            if Obs[i]==k:
                E[m*i+k]=1.0
    G=[]
    for i in range(l):
        G=G+np.random.dirichlet(np.ones(n),size=1).tolist()[0]
    T=list(theta0)
    for i in range(len(rA)):
         for j in range(len(rA[0])):
             T=T+[rA[i,j]]
    for i in range(n):
         for j in range(len(rA[0])):
             T=T+[rB[i,j]]
    Xi=[]
    for t in range(l-1):
            Xi=Xi+np.random.dirichlet(np.ones(n*n),size=1).tolist()[0]
    X=A+B+E+G+T+Xi
    return X
    
def BW_init(theta0,theta,psi,X):
    Obs=X.T.reshape(-1,1)
    n_states=len(theta)
    N=n_states
    n_observations=len(psi[0])
    T=len(X)
    l=T
    A=np.zeros([T,n_states])
    B=np.zeros([T,n_states])
    Xi=np.zeros([T-1,n_states,n_states])
    Xi0=np.zeros([n_states])
    G=np.zeros([T,n_states])
    n=T
    for j in range(n_states):
        A[0,j]=theta0[j]*psi[j,X[0]]
    for t in range(1,n):
        for j in range(n_states):
            A[t,j]=0.0
            for i in range(n_states):
                A[t,j]=A[t,j]+A[t-1,i]*theta[i,j]*psi[j,X[t]]  
    for j in range(n_states):
        B[-1,j]=1.0         
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
    A1=[]
    for i in range(l):
        A1=A1+list((A[i,:])/sum((A[i,:])))
    B1=[]
    for i in range(l-1):
        B1=B1+list((B[i,:])/sum((B[i,:])))
    B1=B1+[1.0]*n_states
    E=[0.0]*(l*n_observations)
    for i in range(l):
        for k in range(n_observations):
            if Obs[i]==k:
                E[n_observations*i+k]=1.0
    G1=[]
    for i in range(l):
        G1=G1+list((G[i,:]))
    T=list(theta0)
    for i in range(N):
        T=T+list(theta[i,:])
    for i in range(N):
        T=T+list(psi[i,:])
    Xi1=[]
    for t in range(l-1):
        for i in range(N):
            Xi1=Xi1+list(Xi[t,i,:])
    #print(len(A1),len(B1),len(E),len(G1),len(T),len(Xi1))
    X=A1+B1+E+G1+T+Xi1       
    return X       
