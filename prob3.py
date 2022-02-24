import pandas as pd
import sklearn as sk
import numpy as np
import math
import statistics

#helper method
def prob3(train,test,d):
        #numc1 and numc2 are counts of classes 1 and 2
        #class 1 has target value 0 and class 2 has target value 1
        numc2=int(train.sum().iat[d])
        n=len(train)
        numc1=n-numc2
        #priorc1 and priorc2 are prior probabilities of classes 1 and 2
        priorc1=numc1/n
        priorc2=numc2/n
        #m1 and m2 are the average vector values of the features of samples in classes 1 and 2
        m1=train.iloc[0:numc1].sum()[:-2]
        m2=train.iloc[numc2:n].sum()[:-2]
        m1=m1/numc1
        m2=m2/numc2
        m1=m1.to_numpy().reshape(d,1)
        m2=m2.to_numpy().reshape(d,1)
        #Sw is the total within class covariance matrix
        Sw=np.zeros((d,d))
        for i in range(numc1):
                row=train.iloc[i,:-2].to_numpy().reshape(d,1)
                row=row-m1
                Sw=np.add(Sw,row.dot(row.transpose()))
        for i in range(numc1,n):
                row=train.iloc[i,:-2].to_numpy().reshape(d,1)
                row=row-m2
                Sw=np.add(Sw,row.dot(row.transpose()))
        Swinv=np.linalg.inv(Sw)
        w=Swinv.dot(np.subtract(m2,m1))
        w=w.transpose()
        #the following line makes w have more normalized component values in most cases
        #it does not affect the math
        w=w*10000
        #c1avgthres and c2avgthres are the average values of data points in the corresponding classes when dotted with w
        #a useful way to think about this is the average threshold associated with a point in each class
        c1avgthres=0
        c2avgthres=0
        for i in range(numc1):
                c1avgthres+=w.dot(train.iloc[i,:-2].to_numpy().reshape(d,1))
        for i in range(numc1,n):
                c2avgthres+=w.dot(train.iloc[i,:-2].to_numpy().reshape(d,1))
        c1avgthres=c1avgthres[0,0]/numc1
        c2avgthres=c2avgthres[0,0]/numc2
        #the following line sets the threshold between c1avgthres and c2avgthres
        #if the prior probability of one class is higher, the threshold is proportionally closer to the other class' average threshold        
        thres=(priorc1*c2avgthres)+(priorc2*c1avgthres)
        #train and test error are the train and test error as percentages
        score=0
        for i in range(n):
                if(w.dot(train.iloc[i,:-2].to_numpy().reshape(d,1))>=thres):
                        #we predict class 1
                        #score increases if we predict class 1 and it is actually class 1
                        score+=train.iat[i,d]
                else:
                        #we predict class 0
                        score+=1
                        #score does not increase if we predict class 0 and it is actually class 1
                        score-=train.iat[i,d]
        error=n-score
        train_error=error/n
        print("error on training set: " + str(train_error))
        score=0
        for i in range(len(test)):
                #same logic as immediately preceding commented code
                if(w.dot(test.iloc[i,:-2].to_numpy().reshape(d,1))>=thres):
                        score+=test.iat[i,d]
                else:
                        score+=1
                        score-=test.iat[i,d]
        error=len(test)-score
        test_error=error/(len(test))
        print("error on test set: " + str(test_error))
        #returning the errors allows them to be appended to the list of errors in LDA1dThres as a single list of two float errors
        return([train_error,test_error])

#this method takes any file and turns the last column into two classes of equal sizes and uses the other columns for prediction
def LDA1dThres(filename,num_crossval):
        errors=[]
        num=num_crossval
        data=pd.read_csv(filename,header=None)
        #d is the number of features
        d=len(data.columns)-1
        data=data.sort_values(by=[d])
        #assigns class 0 to the smallest 50% of target values. assigns one sample to neither class if total number of samples is odd.
        for i in range(math.floor(len(data)/2)):
                data.iat[i,d]=0
                data.iat[i+(math.floor(len(data)/2)),d]=1
        #following line reshuffles data set
        data=data.sample(frac=1)
        data[d+1]=0
        #following loop assigns consecutive samples to consecutive classes
        for i in range(len(data)-1):
                data.iat[i+1,d+1]=(data.iat[i,d+1]+1)%num
        for i in range(num):
                print("training errors for fold " + str(i+1) + ":")
                train=data[data[d+1]!=i].sort_values(by=[d])
                test=data[data[d+1]==i]
                #following line calls helper method prob3 and appends the training and test set errors for one fold to the other errors as a single list containing two elements
                errors.append(prob3(train,test,d))
        #following line stores each set of 
        train_errors,test_errors=map(list,zip(*errors))
        print("average error on training set across all folds: " + str(statistics.mean(train_errors)))
        print("average error on test set across all folds: " + str(statistics.mean(test_errors)))
        print("standard deviation of error on training set across all folds: " + str(statistics.pstdev(train_errors)))
        print("standard deviation of error on test set across all folds: " + str(statistics.pstdev(test_errors)))

print("running LDA1dThres with the boston data set and 10-fold cross validation as arguments")
#this line can be modified to test the method with any other data set without changing anything else
#the method assumes that the last column of any provided data set contains values by which to sort in order to create two classes of equal size
LDA1dThres("boston.csv",10)

