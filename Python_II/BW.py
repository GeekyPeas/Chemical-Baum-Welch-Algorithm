import numpy as np
from hmmlearn import hmm
import csv
import matplotlib.pyplot as plt
import timeit
def BW(theta0,theta,psi,X,it,plot=1):
	#=np.full((1,len(theta)),1/len(theta))[0]
	Obs=X.T.reshape(-1,1)
	n_states=len(theta)
	n_observations=len(psi[0])
	T=len(X)
	A=np.zeros([T,n_states])
	B=np.zeros([T,n_states])
	Xi=np.zeros([T-1,n_states,n_states])
	Xi0=np.zeros([n_states])
	G=np.zeros([T,n_states])
	n=T
	iteration = [0]
	T01=[theta0[0]]
	T02=[theta0[1]]
	T11=[theta[0,0]]
	T12=[theta[0,1]]
	T21=[theta[1,0]]
	T22=[theta[1,1]]
	T1_1=[psi[0,0]]
	T1_2=[psi[0,1]]
	T2_1=[psi[1,0]]
	T2_2=[psi[1,1]]
	for s in range(1,it):	
		for j in range(n_states):
			A[0,j]=theta0[j]*psi[j,X[0]]
		for t in range(1,n):
			for j in range(n_states):
				A[t,j]=0
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
		for i in range(n_states):
			for j in range(n_states): 
				theta[i,j]=sum(Xi[:,i,j])/sum(sum(Xi[:,i,:]))
			for k in range(n_observations):
				psi[i,k]=0
				for t in range(n):
					if k==X[t]:
						psi[i,k]=psi[i,k]+G[t,i]
				psi[i,k]=psi[i,k]/sum(G[:,i])
		iteration.append(s)
		T01.append(theta0[0])
		T02.append(theta0[1])
		T11.append(theta[0,0])
		T12.append(theta[0,1])
		T21.append(theta[1,0])
		T22.append(theta[1,1])
		T1_1.append(psi[0,0])
		T1_2.append(psi[0,1])
		T2_1.append(psi[1,0])
		T2_2.append(psi[1,1])
	model = hmm.CategoricalHMM(n_components=n_states,n_iter=1000,tol=1e-9,init_params=" ",params="te")
	model.startprob_=theta0
	model.transmat_=np.copy(theta)
	model.emissionprob_=np.copy(psi)	
	print(np.log(sum(A[-1,:])))
	print(np.round(model.transmat_,3))
	print(np.round(model.emissionprob_,3))
	print(model.score(Obs))	
	if(plot):
		L1=[]
		L2=[]
		for i in range(len(iteration)):
			T=np.array([[T11[i],T12[i]],
						[T21[i],T22[i]]])
			Psi=np.array([[T1_1[i],T1_2[i]],
						  [T2_1[i],T2_2[i]]])
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

def BWlib(theta0,theta,psi,X,it,plot=1):
	Obs=X.T.reshape(-1,1)
	n_states=len(theta)
	n_observations=len(psi[0])
	T=len(X)
	n=T
	iteration = [0]
	T01=[theta0[0]]
	T02=[theta0[1]]
	T11=[theta[0,0]]
	T12=[theta[0,1]]
	T21=[theta[1,0]]
	T22=[theta[1,1]]
	T1_1=[psi[0,0]]
	T1_2=[psi[0,1]]
	T2_1=[psi[1,0]]
	T2_2=[psi[1,1]]
	model=hmm.CategoricalHMM(n_components=n_states,n_iter=1,tol=1e-9,init_params=" ",params="te")
	model.startprob_=theta0
	model.transmat_=theta
	model.emissionprob_=psi			
	for s in range(1,it):
		model.fit(X)
		iteration.append(s)
		T01.append(model.startprob_[0])
		T02.append(model.startprob_[1])
		T11.append(model.transmat_[0,0])
		T12.append(model.transmat_[0,1])
		T21.append(model.transmat_[1,0])
		T22.append(model.transmat_[1,1])
		T1_1.append(model.emissionprob_[0,0])
		T1_2.append(model.emissionprob_[0,1])
		T2_1.append(model.emissionprob_[1,0])
		T2_2.append(model.emissionprob_[1,1])
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
			T=np.array([[T11[i],T12[i]],
						[T21[i],T22[i]]])
			Psi=np.array([[T1_1[i],T1_2[i]],
						  [T2_1[i],T2_2[i]]])
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
def crn_Liklihood(filenamecsv,start,Obs):
	model = hmm.CategoricalHMM(n_components=2,n_iter=1,tol=1e-9,init_params=" ",params="te")
	t = []
	T01=[]
	T02=[]
	T11=[]
	T12=[]
	T21=[]
	T22=[]
	T1_1=[]
	T1_2=[]
	T2_1=[]
	T2_2=[]
	c=start
	with open(filenamecsv,'r') as csvfile:
		#next(csvfile)
		plots = csv.reader(csvfile, delimiter=' ')
		for row in plots:
			t.append(float(row[0]))
			T01.append(float(row[c+1]))
			T02.append(float(row[c+2]))
			T11.append(float(row[c+3]))
			T12.append(float(row[c+4]))
			T21.append(float(row[c+5]))
			T22.append(float(row[c+6]))
			T1_1.append(float(row[c+7]))
			T1_2.append(float(row[c+8]))
			T2_1.append(float(row[c+9]))
			T2_2.append(float(row[c+10]))
		
	L1=[]
	L2=[]
	for i in range(len(t)):
		T0=np.array([T01[i],T02[i]])
		T=np.array([[T11[i],T12[i]],
			    [T21[i],T22[i]]])
		Psi=np.array([[T1_1[i],T1_2[i]],
		              [T2_1[i],T2_2[i]]])
		T=np.absolute(T)
		Psi=np.absolute(Psi)
		model.startprob_=T0
		model.transmat_=T
		model.emissionprob_=Psi
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
	return t,L1,T01,T02,T11,T12,T21,T22,T1_1,T1_2,T2_1,T2_2
def print_conc(start,Y1,Y2):
	I=np.array(Y1).astype(np.float)
	spt0=np.array([I[start],I[start+1]])
	spt=np.array([[I[start+2],I[start+3]],[I[start+4],I[start+5]]])
	spe=np.array([[I[start+6],I[start+7]],[I[start+8],I[start+9]]])
	print("CRN initialization")
	print("Starting Prob:\n",spt0)
	print("Transition:\n",spt)
	print("Emission:\n",spe)
	I=np.array(Y2).astype(np.float)
	print("\nCRN Equilibrium")
	pt0=np.round(np.array([I[start],I[start+1]]),3)
	pt=np.round(np.array([[I[start+2],I[start+3]],[I[start+4],I[start+5]]]),3)
	pe=np.round(np.array([[I[start+6],I[start+7]],[I[start+8],I[start+9]]]),3)
	print("Starting Prob:\n",pt0)
	print("Transition:\n",pt)
	print("Emission:\n",pe)
def print_conc_rounded(start,Y1,Y2):
	I=np.array(Y1).astype(np.float)
	spt0=np.round(np.array([I[start],I[start+1]]),3)
	spt=np.round(np.array([[I[start+2],I[start+3]],[I[start+4],I[start+5]]]),3)
	spe=np.round(np.array([[I[start+6],I[start+7]],[I[start+8],I[start+9]]]),3)
	print("CRN initialization")
	print("Starting Prob:\n",spt0)
	print("Transition:\n",spt)
	print("Emission:\n",spe)
	I=np.array(Y2).astype(np.float)
	print("\nCRN Equilibrium")
	pt0=np.round(np.array([I[start],I[start+1]]),3)
	pt=np.round(np.array([[I[start+2],I[start+3]],[I[start+4],I[start+5]]]),3)
	pe=np.round(np.array([[I[start+6],I[start+7]],[I[start+8],I[start+9]]]),3)
	print("Starting Prob:\n",pt0)
	print("Transition:\n",pt)
	print("Emission:\n",pe)
def crn_Liklihoodp(filenamecsv,start,Obs):
	model = hmm.CategoricalHMM(n_components=2,n_iter=1,tol=1e-9,init_params=" ",params="te")
	t = []
	T01=[]
	T02=[]
	T11=[]
	T12=[]
	T21=[]
	T22=[]
	T1_1=[]
	T1_2=[]
	T2_1=[]
	T2_2=[]
	c=start
	with open(filenamecsv,'r') as csvfile:
		#next(csvfile)
		plots = csv.reader(csvfile, delimiter=' ')
		for row in plots:
			t.append(float(row[0]))
			T01.append(float(row[c+1]))
			T02.append(float(row[c+2]))
			T11.append(float(row[c+3]))
			T12.append(float(row[c+4]))
			T21.append(float(row[c+5]))
			T22.append(float(row[c+6]))
			T1_1.append(float(row[c+7]))
			T1_2.append(float(row[c+8]))
			T2_1.append(float(row[c+9]))
			T2_2.append(float(row[c+10]))
		
	L1=[]
	L2=[]
	for i in range(len(t)):
		T0=np.array([T01[i],T02[i]])
		T=np.array([[T11[i],T12[i]],
			    [T21[i],T22[i]]])
		Psi=np.array([[T1_1[i],T1_2[i]],
		              [T2_1[i],T2_2[i]]])
		T=np.absolute(T)
		Psi=np.absolute(Psi)
		model.startprob_=T0
		model.transmat_=T
		model.emissionprob_=Psi
		L1.append(model.score(Obs))
		L2.append(model.decode(Obs, algorithm="viterbi")[0])
	#print(model.score(Obs))
	#print("CRN_Transition fit:\n",np.round(model.transmat_,3))
	#print("True_Transition:\n",transition_probability)
	#print(" ")
	#print("CRN_Emission fit:\n",np.round(model.emissionprob_,3))
	#print("True_Emission:\n",emission_probability)
	return t,L1,T01,T02,T11,T12,T21,T22,T1_1,T1_2,T2_1,T2_2  
