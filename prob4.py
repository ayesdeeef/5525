import pandas as pd
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
import scipy as sp

plt.close("all")

#This method is Naive Bayes'
def prob4NB(train,test,d,k):
	n=len(train)
	#counts is the number of samples in each class
	counts=[0]*k
	#priors is the fraction of samples in each class
	priors=[0]*k
	#mus is the mean within each class
	mus=[0]*k
	#covs is the covariance within each class
	covs=[0]*k
	#ps is a monotonically increasing function of the likelihood of a class given a sample for each class
	ps=[0]*k
	for i in range(n):
		counts[train.iloc[i,d]]+=1
		mus[train.iloc[i,d]]+=train.iloc[i,:d].to_numpy().reshape(d,1)
	for i in range(k):
		priors[i]=counts[i]/n
		mus[i]=mus[i]/counts[i]
	for i in range(n):
		x=train.iloc[i,:d].to_numpy().reshape(d,1)
		diff=x-mus[train.iloc[i,d]]
		covs[train.iloc[i,d]]+=diff.dot(diff.transpose())
	for i in range(k):
		covs[i]=covs[i]/counts[i]
	for i in range(len(test)):
		x=test.iloc[i,:d].to_numpy().reshape(d,1)
		for j in range(k):
			ps[j]=np.log(priors[j])+(d/2)*np.log(1/(2*math.pi))-(1/2)*np.log(np.linalg.det(covs[j]))-(1/2)*(np.transpose(x-mus[j]).dot(np.linalg.pinv(covs[j]))).dot(x-mus[j])
		test.iloc[i,d+1]=ps.index(max(ps))
	score=0
	for i in range(len(test)):
		#error rate calculation as in problem 3
		if(test.iloc[i,d]==test.iloc[i,d+1]):
			score+=1
	error=len(test)-score
	test_error=error/(len(test))
	#returning the error allows it to be appended to errorsel in main
	print(test_error)
	return(test_error)

#This method is Logistic Regression
def prob4LR(train,test,d,k):
	n=len(train)
	#one-hot encoding would look like the following four lines but I'm not going to use them:
#	for i in range(k):
#		train[d+1+k]=0
#	for i in range(n):
#		train.iat[i,d+1+train.iat[i,d]]=1
	#Initializing
	w=np.zeros((d,1))
	thres=0
	for i in range(500):
		#using mini-batch gradient descent
		for j in range(math.floor((n-1)/100)+1):
			first=100*j
			last=100*(j+1)
			x=train.iloc[first:last,:d].to_numpy().reshape(len(train.iloc[first:last]),d)
			y=train.iloc[first:last,d].to_numpy().reshape(len(train.iloc[first:last]),1)
			#x is 100 by d and w is d by 1
			#thres is a scalar
			#z is 100 by 1
			z=np.dot(x,w)+thres
			#using sigmoid function (giving up on softmax and multiclass classification for now)
			#y_hat is 100 by 1, the predictions on the current batch
			y_hat=1./(1.+np.exp(-1.*z))
			dw=np.dot(x.transpose(),y_hat-y)/100
			dthres=np.sum((y_hat-y))/100
			w-=0.01*dw
			thres-=0.01*dthres
	z=np.dot(test.iloc[:,:d].to_numpy().reshape(len(test),d),w)+thres
	#predictions go in the d+1 column
	test[d+1]=1./(1.+np.exp(-1.*z))
	test[d+1]=test[d+1].round()
	test[d+1]=test[d+1].astype(int)
	score=0
	for i in range(len(test)):
		#error rate calculation as in problem 3
		if(test.iloc[i,d]==test.iloc[i,d+1]):
			score+=1
	error=len(test)-score
	test_error=error/(len(test))
	#returning the error allows it to be appended to errorsel in main
	print(test_error)
	return(test_error)

def prob4(file,alg,k):
	if(file=="boston.csv"):
		boston(alg,k)
	else:
		normal(file,alg,k)
	plt.show()

#the following method is for the boston dataset and executes both boston50 and boston75 simultaneously
def boston(alg,k):
	data=pd.read_csv("boston.csv",header=None)
	n=len(data)
	d=len(data.columns)-1
	boston50=data.sort_values(by=[d])
#assigns class 0 to the smallest 50% of target values. assigns last sample to neither class if total number of samples is odd.
	for i in range(math.floor(n/2)):
		boston50.iat[i,d]=0
		boston50.iat[i+(math.floor(n/2)),d]=1
	boston50[d]=boston50[d].astype(int)
	boston75=data.sort_values(by=[d])
	for i in range(math.floor(n*3/4)):
		boston75.iat[i,d]=0
	for i in range(math.floor(n*3/4),n):
		boston75.iat[i,d]=1
	boston75[d]=boston75[d].astype(int)
	print("testing output for boston50 data set")
	runner(boston50,alg,k)
	print("testing output for boston75 data set")
	runner(boston75,alg,k)

#the following method is for any dataset that is not the boston dataset
def normal(file,alg,k):
	print("testing output for " + file + " data set")
	data=pd.read_csv(file,header=None)
	runner(data,alg,k)

#this method calls the appropriate classification algorithm, Naive Bayes' or Logistic Regression
def runner(data,alg,k):
	n=len(data)
	d=len(data.columns)-1
	dt=data.copy()
	means=[0]*5
	stdevs=[0]*5
	lst=[10,25,50,75,100]
	errors10=[0]*10
	errors25=[0]*10
	errors50=[0]*10
	errors75=[0]*10
	errors100=[0]*10
	errors={10:errors10,25:errors25,50:errors50,75:errors75,100:errors100}
	for i in range(10):
		print("sample errors for training/test split number " + str(i+1) + ":")
		dt[d+1]=1
		dt=dt.sample(frac=1)
		for j in range(n-1):
			dt.iat[j+1,d+1]=dt.iat[j,d+1]+1
		train=dt[dt[d+1]<=math.ceil(n*4/5)]
		test=dt[dt[d+1]>math.ceil(n*4/5)]
		for j in range(5):
			trainel=train.sample(frac=lst[j]/100)
			if(alg=="NB"):
				print("error when using " + str(lst[j]) + "% of training set for training:")
				errors[lst[j]][i]=prob4NB(trainel,test,d,k)
			elif(alg=="LR"):
				errors[lst[j]][i]=prob4LR(trainel,test,d,k)
			else:
				print("invalid algorithm selection. choose NB or LR, enterred as a string")
	for i in range(5):
		means[i]=statistics.mean(errors[lst[i]])
		stdevs[i]=statistics.pstdev(errors[lst[i]])
	plt.errorbar(lst,means,yerr=stdevs)
	for i in range(5):
		print("average error across all ten repetitions when using " + str(lst[i]) + "% of training set for training: " + str(means[i]))
		print("standard deviation of error across all ten repetitions when using " + str(lst[i]) + "% of training set for training: " + str(stdevs[i]))

###comment all but one of the following lines to see the corresponding output and plot
###NB doesn't give meaningful results for the digits dataset only because I wasn't able to find the right workarounds for certain computational limitations in Python (I think)
###LR does not work at all for the digits dataset because I was not able to put together a serious attempt at multiclass classification with logistic regression
prob4("boston.csv","NB",2)
prob4("digits.csv","NB",10)
prob4("boston.csv","LR",2)
prob4("digits.csv","LR",10)
