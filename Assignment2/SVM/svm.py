
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import math
import sys
import pickle
import time
from cvxopt import matrix
import statistics
from cvxopt import solvers
import seaborn as sn
from collections import Counter
solvers.options['show_progress'] = False
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


# In[4]:


train=pd.read_csv("train.csv",header=None).to_numpy()
test=pd.read_csv("test.csv",header=None).to_numpy()
validate=pd.read_csv("val.csv",header=None).to_numpy()


# <font size="3.5">__Q1 Part A Linear Kernel through CVXOPT__</font>

# In[133]:


def mainLinear(C,class1,class2):
    
    trainTemp=train[(train[:,-1]==class1) | (train[:,-1]==class2)].copy()#[:,:-1]/255
    validateTemp=validate[(validate[:,-1]==class1) | (validate[:,-1]==class2)].copy()#[:,:-1]/255
    testTemp=test[(test[:,-1]==class1) | (test[:,-1]==class2)].copy()#[:,:-1]/255
    
    positiveTrain=trainTemp[(trainTemp[:,-1]==class1)].copy()[:,:-1]/255
    negativeTrain=trainTemp[(trainTemp[:,-1]==class2)].copy()[:,:-1]/255
    
    trainClassOut = trainTemp[:,-1].copy().reshape(-1,1)
    validateClassOut = validateTemp[:,-1].copy().reshape(-1,1)
    testClassOut = testTemp[:,-1].copy().reshape(-1,1)
    
    
    posIndTrain,negIndTrain=np.where(trainClassOut==class1),np.where(trainClassOut==class2)
    posIndTest,negIndTest=np.where(testClassOut==class1),np.where(testClassOut==class2)
    posIndVali,negIndVali=np.where(validateClassOut==class1),np.where(validateClassOut==class2)
    
    
    trainClassOut[negIndTrain] = -1
    trainClassOut[posIndTrain] = 1 
    
    
    testClassOut[posIndTest],testClassOut[negIndTest]=1,-1
    validateClassOut[posIndVali],validateClassOut[negIndVali]=1,-1
   
    trainClass,testClass,validateClass=trainTemp[:,:-1]/255,testTemp[:,:-1]/255,validateTemp[:,:-1]/255
    
    
    def findAttributes(c):
        m=trainClass.shape[0]
        q=-np.ones((m,1))
        a=trainClassOut.T
        b=np.zeros((1,1))
        x_Mult=np.dot(trainClass,trainClass.T)
        y_Mult=np.dot(trainClassOut,trainClassOut.T)
        p=np.multiply(x_Mult,y_Mult) #Calculated the value of p
        g=np.append(np.diag(np.ones(m)),np.diag(-np.ones(m)),axis=0)
        h=np.append(np.full((m,1),c),np.zeros((m,1))).reshape(2*m,1)

        return p,q,g,h,a,b
    
    mystart=time.time()
    
    def training():
        p,q,g,h,a,b=findAttributes(C)
        P=matrix(p,tc='d')
        Q=matrix(q,tc='d')
        G=matrix(g,tc='d')
        H=matrix(h,tc='d')
        A=matrix(a,tc='d')
        B=matrix(b,tc='d')
        solution=solvers.qp(P,Q,G,H,A,B)
        return solution
    
    solution=training()
    
    #Taking the threshold as 1e-5
    #This function will return the support vectors, it takes solution as an attribute. Work for both linear and gaussian kernel.
    def support_vectors(solPara):
        alpha = np.array(solPara['x'])
        
        print("Number of support vectors are ",sum(alpha>1e-5))
        
        inputWithAlpha=np.append(trainClass,alpha,axis=1)
        inputAlphaOut=np.append(inputWithAlpha,trainClassOut,axis=1) #Here we have combined input with alpha and the corresponding output

        combinedVectors=inputAlphaOut[(inputAlphaOut[:,-2]>1e-5)]
    #     supportVectors=combinedVectors[:,:-1]
        out=combinedVectors
        print("what is out here in support vector function")
        print(out)
        return out

    #WE are returning whole output input with alpha with output and then will slice acc to need.
    outputFromSupport=support_vectors(solution)
    
    #score from here, we would get our desired support vector
    supportVector=outputFromSupport[:,:-2]
    
    # print(supportVector)
    def findWAndB(): 
        #Here, we have taken outputFromSupport 
        alpha=outputFromSupport[:,-2]
        alpha=alpha.reshape(alpha.size,1)
        y=outputFromSupport[:,-1]
        y=y.reshape(y.size,1)
        x=outputFromSupport[:,:-2]
        temp=np.multiply(np.multiply(alpha,y),x)
        out=temp[0]

        for vector in temp[1:,:]:
    #         print(vector.shape)
            out=np.add(out,vector)
        #Now here comes the part for b
    
        wHere=out.reshape(out.size,1) #Taking w for dot product of w and x , w^T * x
        positiveOut=min(np.dot(positiveTrain,wHere))
        negativeOut=max(np.dot(negativeTrain,wHere))
        bHere=-(positiveOut+negativeOut)/2
        return out,bHere
    

    w,b=findWAndB()
    print("Time my implementation took for linear svm is ", time.time()-mystart)
    print(w.shape)
    print(b)
    
    def accuracy():
        valiProd=np.dot(validateClass,w)
        testProd=np.dot(testClass,w)
        trainProd=np.dot(trainClass,w)
        valiOut=valiProd+b
        testOut=testProd+b
        trainOut=trainProd+b
        valiFinal=np.array([1 if item>=0 else -1 for item in valiOut]).reshape(-1,1)
        testFinal=np.array([1 if item>=0 else -1 for item in testOut]).reshape(-1,1)
        trainFinal=np.array([1 if item>=0 else -1 for item in trainOut]).reshape(-1,1)
        print("Validate Accuracy is ",(np.sum(valiFinal==validateClassOut)/valiFinal.size)*100," % ")
        print("Test Accuracy is ",(np.sum(testFinal==testClassOut)/testFinal.size)*100," % ")
        print("Train Accuracy is ",(np.sum(trainFinal==trainClassOut)/trainFinal.size)*100," % ")
         
    accuracy()    


# In[134]:


mainLinear(1,0,9)


# <font size="3.5">__Q1 Part B Gaussian Kernel through CVXOPT__</font>

# In[131]:


def mainGaussian(C,Gamma,class1,class2):
    dummy=time.time()
    
    trainTemp=train[(train[:,-1]==class1) | (train[:,-1]==class2)].copy()#[:,:-1]/255
    validateTemp=validate[(validate[:,-1]==class1) | (validate[:,-1]==class2)].copy()#[:,:-1]/255
    testTemp=test[(test[:,-1]==class1) | (test[:,-1]==class2)].copy()#[:,:-1]/255
    
    positiveTrain=trainTemp[(trainTemp[:,-1]==class1)][:,:-1]/255
    negativeTrain=trainTemp[(trainTemp[:,-1]==class2)][:,:-1]/255
    
    trainClassOut = trainTemp[:,-1].copy().reshape(-1,1)
    validateClassOut = validateTemp[:,-1].copy().reshape(-1,1)
    testClassOut = testTemp[:,-1].copy().reshape(-1,1)
    
    
    posIndTrain,negIndTrain=np.where(trainClassOut==class1),np.where(trainClassOut==class2)
    posIndTest,negIndTest=np.where(testClassOut==class1),np.where(testClassOut==class2)
    posIndVali,negIndVali=np.where(validateClassOut==class1),np.where(validateClassOut==class2)
    
    
    trainClassOut[negIndTrain] = -1
    trainClassOut[posIndTrain] = 1 
    
    
    testClassOut[posIndTest],testClassOut[negIndTest]=1,-1
    validateClassOut[posIndVali],validateClassOut[negIndVali]=1,-1
   
    trainClass,testClass,validateClass=trainTemp[:,:-1]/255,testTemp[:,:-1]/255,validateTemp[:,:-1]/255

    def gaussianAttributes(c,gamma):
        m=trainClass.shape[0]

        q=-np.ones((m,1))
        a=trainClassOut.T
        b=np.zeros((1,1))
        def findx_Mult():

            temp=np.exp((distance.cdist(trainClass,trainClass)**2)*(-1*gamma))
            return temp

        x_Mult=findx_Mult()

        y_Mult=np.dot(trainClassOut,trainClassOut.T)
        p=np.multiply(x_Mult,y_Mult) #Calculated the value of p
        g=np.append(np.diag(np.ones(m)),np.diag(-np.ones(m)),axis=0)
        h=np.append(np.full((m,1),c),np.zeros((m,1))).reshape(2*m,1)

        return p,q,g,h,a,b #For gaussian parameters
    
        # Gamma = 0.05 , c=1.0
    start=time.time()
    def gaussianTraining():
        p,q,g,h,a,b=gaussianAttributes(C,Gamma)

        P,Q,G,H,A,B=matrix(p,tc='d'),matrix(q,tc='d'),matrix(g,tc='d'),matrix(h,tc='d'),matrix(a,tc='d'),matrix(b,tc='d')
        solution=solvers.qp(P,Q,G,H,A,B)
        return solution
    
    gaussianSolution=gaussianTraining()
    
    #Taking the threshold as 1e-5
    #This function will return the support vectors, it takes solution as an attribute. Work for both linear and gaussian kernel.
    def support_vectors(solPara):
        alpha = np.array(solPara['x'])
#         print("alpha kya hai")
        print("Number of support vectors are",sum(alpha>1e-5))

        inputWithAlpha=np.append(trainClass,alpha,axis=1)
#         print(inputWithAlpha.shape)
        inputAlphaOut=np.append(inputWithAlpha,trainClassOut,axis=1) #Here we have combined input with alpha and the corresponding output
#         print(inputAlphaOut.shape)
        combinedVectors=inputAlphaOut[inputAlphaOut[:,-2]>1e-5]

        out=combinedVectors

        return out
    
    def gaussianSupportVectors():
        gaussianAttrFromSupport=support_vectors(gaussianSolution)
        gaussianSupportVector=gaussianAttrFromSupport[:,:-2] #It will give support vector for gaussian
        gaussianAlpha=gaussianAttrFromSupport[:,-2]
        # For number of gaussian support vectors
#         print("Number of gaussian support vectors are ", sum(gaussianAlpha>1e-5)) 
        return gaussianSupportVector
    
    gaussianSupportVector=gaussianSupportVectors()
#     print(gaussianSupportVector)
    
    def gaussianReturnParamters(gamma):
        #Here we would find b and accuracy thereby
        gaussianOutFromSupport=support_vectors(gaussianSolution)
        y=gaussianOutFromSupport[:,-1]
        alpha=gaussianOutFromSupport[:,-2]
        x=gaussianOutFromSupport[:,:-2]


        def findB():
            #For Positive Max
            posXOut,negXOut=[],[]
            #isko cdist se handle krna hai

            temp1=np.exp((distance.cdist(positiveTrain,x)**2)*-gamma)*(alpha.reshape(1,-1))*(y.reshape(1,-1))
#             print("temp1 is ",temp1)
            minPos=min(temp1.sum(axis=1))
            temp2=np.exp((distance.cdist(negativeTrain,x)**2)*-gamma)*(alpha.reshape(1,-1))*(y.reshape(1,-1))
#             print("temp2 is ",temp2)
            maxNeg=max(temp2.sum(axis=1))
            

            
            b=-(minPos+maxNeg)/2
            return b

        b=findB()
        print("Total time took for gaussian training is ",time.time()-start)
        
        return gaussianOutFromSupport,b
    alphaAndY,bFinal=gaussianReturnParamters(Gamma)
    
    return alphaAndY,bFinal


# In[140]:


def gaussianAccuracy(C,class1,class2,gamma):
    
    trainTemp=train[(train[:,-1]==class1) | (train[:,-1]==class2)].copy()#[:,:-1]/255
    validateTemp=validate[(validate[:,-1]==class1) | (validate[:,-1]==class2)].copy()#[:,:-1]/255
    testTemp=test[(test[:,-1]==class1) | (test[:,-1]==class2)].copy()#[:,:-1]/255
    
    positiveTrain=trainTemp[(trainTemp[:,-1]==class1)][:,:-1]/255
    negativeTrain=trainTemp[(trainTemp[:,-1]==class2)][:,:-1]/255
    
    trainClassOut = trainTemp[:,-1].copy().reshape(-1,1)
    validateClassOut = validateTemp[:,-1].copy().reshape(-1,1)
    testClassOut = testTemp[:,-1].copy().reshape(-1,1)
    
    
    posIndTrain,negIndTrain=np.where(trainClassOut==class1),np.where(trainClassOut==class2)
    posIndTest,negIndTest=np.where(testClassOut==class1),np.where(testClassOut==class2)
    posIndVali,negIndVali=np.where(validateClassOut==class1),np.where(validateClassOut==class2)
    
    
    trainClassOut[negIndTrain] = -1
    trainClassOut[posIndTrain] = 1 
    
    
    testClassOut[posIndTest],testClassOut[negIndTest]=1,-1
    validateClassOut[posIndVali],validateClassOut[negIndVali]=1,-1
   
    trainClass,testClass,validateClass=trainTemp[:,:-1]/255,testTemp[:,:-1]/255,validateTemp[:,:-1]/255
    
    alphaAndY,b=mainGaussian(C,gamma,class1,class2)
    x,y,alpha=alphaAndY[:,:-2],alphaAndY[:,-1],alphaAndY[:,-2]
    
    print("value of b in gaussian is ",b)
    
    #Now we will use this b to calculate accuracy
    trainPredicted,testPredicted,validatePredicted=[],[],[]

    temp1=((np.exp((distance.cdist(trainClass,x)**2)*-gamma)*(alpha.reshape(1,-1))*(y.reshape(1,-1))).sum(axis=1))+b
    temp2=((np.exp((distance.cdist(testClass,x)**2)*-gamma)*(alpha.reshape(1,-1))*(y.reshape(1,-1))).sum(axis=1))+b
    temp3=((np.exp((distance.cdist(validateClass,x)**2)*-gamma)*(alpha.reshape(1,-1))*(y.reshape(1,-1))).sum(axis=1))+b

    trainPredicted=np.array([1 if item>=0 else -1 for item in temp1]).reshape(-1,1)
    testPredicted=np.array([1 if item>=0 else -1 for item in temp2]).reshape(-1,1)
    validatePredicted=np.array([1 if item>=0 else -1 for item in temp3]).reshape(-1,1)

    #Now here we will find the gaussian accuracy
    print("Validate Accuracy through gaussian kernel is ",(np.sum(validatePredicted==validateClassOut)/validatePredicted.size)*100," % ")
    print("Test Accuracy through gaussian kernel is ",(np.sum(testPredicted==testClassOut)/testPredicted.size)*100," % ")
    print("Train Accuracy through gaussian kernel is ",(np.sum(trainPredicted==trainClassOut)/trainPredicted.size)*100," % ")

gaussianAccuracy(1.0,9,0,0.05)  


# <font size="3.5">__Q1 Part C Linear & Gaussian Kernel through skLearn__</font>

# In[104]:


def skLearn(cPara,gammaPara,class1,class2,kernelPara):
    
    trainTemp=train[(train[:,-1]==class1) | (train[:,-1]==class2)].copy()#[:,:-1]/255
    validateTemp=validate[(validate[:,-1]==class1) | (validate[:,-1]==class2)].copy()#[:,:-1]/255
    testTemp=test[(test[:,-1]==class1) | (test[:,-1]==class2)].copy()#[:,:-1]/255
    
    trainClassOut = trainTemp[:,-1].copy().reshape(-1,1)
    validateClassOut = validateTemp[:,-1].copy().reshape(-1,1)
    testClassOut = testTemp[:,-1].copy().reshape(-1,1)
    
    trainClass,testClass,validateClass=trainTemp[:,:-1]/255,testTemp[:,:-1]/255,validateTemp[:,:-1]/255
    
#     trainClassOut[trainClassOut==class2]=-1
#     validateClassOut[validateClassOut==class2]=-1
#     testClassOut[testClassOut==class2]=-1
#     trainClassOut[trainClassOut==class1]=1
#     validateClassOut[validateClassOut==class1]=1
#     testClassOut[testClassOut==class1]=1
    
    start=time.time()
    
    if kernelPara=="linear":
        model=SVC(C=cPara,kernel=kernelPara)
    else:
        model=SVC(C=cPara,gamma=gammaPara,kernel=kernelPara)
        
    model.fit(trainClass,trainClassOut.ravel())

    print("Time sklearn svm took is ",time.time()-start)
    
    return model


# In[105]:


#This part is for finding gaussian accuracy after the above function returns parameter
#In question i had to predict for class 9 and 0

def printingsKLrnAccuracies(c,gamma,class1,class2,kernel):

    trainTemp=train[(train[:,-1]==class1) | (train[:,-1]==class2)].copy()#[:,:-1]/255
    validateTemp=validate[(validate[:,-1]==class1) | (validate[:,-1]==class2)].copy()#[:,:-1]/255
    testTemp=test[(test[:,-1]==class1) | (test[:,-1]==class2)].copy()#[:,:-1]/255
    
    trainClassOut = trainTemp[:,-1].copy().reshape(-1,1)
    validateClassOut = validateTemp[:,-1].copy().reshape(-1,1)
    testClassOut = testTemp[:,-1].copy().reshape(-1,1)
    
    trainClass,testClass,validateClass=trainTemp[:,:-1]/255,testTemp[:,:-1]/255,validateTemp[:,:-1]/255

# We will uncomment below things to get value of b else it is working just fine.    
#     trainClassOut[trainClassOut==class2]=-1
#     validateClassOut[validateClassOut==class2]=-1
#     testClassOut[testClassOut==class2]=-1
#     trainClassOut[trainClassOut==class1]=1
#     validateClassOut[validateClassOut==class1]=1
#     testClassOut[testClassOut==class1]=1
    
    model=skLearn(c,gamma,class1,class2,kernel)
    trainPredicted,testPredicted,validatePredicted=model.predict(trainClass),model.predict(testClass),model.predict(validateClass)
    
# Commenting these we are generalizing the code to be used in Q2 Part B
#     print('w = ',model.coef_)
    print("Number of support vectors are ",model.n_support_)
    print('b = ',model.intercept_)
    print("Train Output through skLearn svm is ", accuracy_score(trainClassOut,trainPredicted)*100," %")
    print("Test Output through skLearn svm is ", accuracy_score(testClassOut,testPredicted)*100," %")
    print("Validate Output through skLearn svm is ", accuracy_score(validateClassOut,validatePredicted)*100," %")

#Fourth attribute is for type of kernel, "linear" for linear kernel, "rbf" for gaussian kernel
# skLearn(1.0,0.05,9,0,"linear")
printingsKLrnAccuracies(1.0,0.05,0,9,"linear")


# __Time sklearn linear svm took is  0.22699189186096191__
# 
# __Number of support vectors for linear svm are 57__
# 
# __Value of b for linear svm is -0.76384274__
# 
# __Test Accuracy, Validate Accuracy obtained are 100.0% , 99.8%__
# 
# __Time sklearn Gaussian svm took is 3.4515771865844727__
# 
# __Number of support vectors for gaussian svm are 826__
# 
# __Value of b for gaussian svm is -0.18758554__
# 
# __Test Accuracy, Validate Accuracy obtained are 99.8% , 100%__

# # Multi Class Classification

# In[ ]:


def multiClassCVXOPT(cPara,gammaPara):
    
    xAlphaY,b=[],[]
    
    #Here, we are training the model
    
    for class1 in tqdm(range(10)):
        for class2 in tqdm(range(class1+1,10)):
            xAlphaYTemp,bTemp=mainGaussian(cPara,gammaPara,class1,class2)
            xAlphaY.append(xAlphaYTemp)
            b.append(bTemp)
               
#     with open('xAlphaY','wb') as f:
#         pickle.dump(xAlphaY,f)
        
#     with open('bCVXOTP','wb') as f:
#         pickle.dump(b,f)


# In[ ]:


multiClassCVXOPT(1.0,0.05)


# In[6]:


#After we got our required parameters from our learned model
def multiClassPrediction(C,gamma):
    
#     with open('xAlphaY','rb') as f:
#         xAlphaY=pickle.load(f)
        
#     with open('bCVXOTP','rb') as f:
#         b=pickle.load(f)
    
    xAlphaY=np.asarray(xAlphaY)

    print("xAlphaY ka shape ")
    print(xAlphaY.shape)
    
    trainClass=train[:,:-1]/255
    validateClass=validate[:,:-1]/255
    testClass=test[:,:-1]/255
    trainClassOut=train[:,-1].reshape(-1,1)
    validateClassOut=validate[:,-1].reshape(-1,1)
    testClassOut=test[:,-1].reshape(-1,1)    
    
    validatePredicted,testPredicted,trainPredicted,trainScore,testScore,validateScore=[],[],[],[],[],[]
    for (xAY,B) in tqdm(zip(xAlphaY,b)):

        x,y,alpha=xAY[:,:-2],xAY[:,-1],xAY[:,-2]
        
        
#         temp1=((np.exp((distance.cdist(trainClass,x)**2)*-gamma)*(alpha.reshape(1,-1))*(y.reshape(1,-1))).sum(axis=1))+B
        temp2=((np.exp((distance.cdist(testClass,x)**2)*-gamma)*(alpha.reshape(1,-1))*(y.reshape(1,-1))).sum(axis=1))+B
        temp3=((np.exp((distance.cdist(validateClass,x)**2)*-gamma)*(alpha.reshape(1,-1))*(y.reshape(1,-1))).sum(axis=1))+B
#         trainScore.append(temp1)
        testScore.append(temp2)
        validateScore.append(temp3)
#         break
    k=0
    
#     trainSumScore=np.zeros((trainClass.shape[0],10))
    testSumScore=np.zeros((testClass.shape[0],10))
    validateSumScore=np.zeros((validateClass.shape[0],10))
    
    
    for i in tqdm(range(10)):
        for j in range(i+1,10):    
#             trainPredicted.append([i if item>=0 else j for item in trainScore[k]])
            testPredicted.append([i if item>=0 else j for item in testScore[k]])
            validatePredicted.append([i if item>=0 else j for item in validateScore[k]])
            k+=1
#             break
#         break    
        
    #Score to sum krna padega na re
    trainScore,testScore,validateScore=np.array(trainScore).T,np.array(testScore).T,np.array(validateScore).T 
    trainPredicted,testPredicted,validatePredicted=np.array(trainPredicted).T,np.array(testPredicted).T,np.array(validatePredicted).T
    
#     for i in range(len(trainPredicted)):
#         for j in range(len(trainPredicted[i])):
#             trainSumScore[i,trainPredicted[i,j]]+=abs(trainScore[i,j])
            
    for i in range(len(testPredicted)):
        for j in range(len(testPredicted[i])):
            testSumScore[i,testPredicted[i,j]]+=abs(testScore[i,j])
    
    for i in range(len(validatePredicted)):
        for j in range(len(validatePredicted[i])):
            validateSumScore[i,validatePredicted[i,j]]+=abs(validateScore[i,j])
            
    #yahan saare score store kar liye
    
#     print("printing trainPredicted for trainPredicted")
#     print(trainPredicted)
    
    trainFinal,validateFinal,testFinal=[],[],[]

#     for i in tqdm(range(len(trainPredicted))):
        
#         highest,itemsWithHighFrequency=np.argmax(np.bincount(trainPredicted[i])),[]
#         frequency = Counter(trainPredicted[i])
#         value = frequency[highest]
#         for element,occurence in frequency.items():
#             if occurence==value:
#                 itemsWithHighFrequency.append(element)
        
#         if len(itemsWithHighFrequency)==1:
#             trainFinal.append(itemsWithHighFrequency[0])
#         else:
#             maxm,maxmScore=itemsWithHighFrequency[0],trainSumScore[i,itemsWithHighFrequency[0]]
#             for l in range(1,len(itemsWithHighFrequency)):
#                 if trainSumScore[i,itemsWithHighFrequency[l]] > maxmScore:
#                     maxmScore = trainSumScore[i,itemsWithHighFrequency[l]]
#                     maxm = itemsWithHighFrequency[l]
                    
#             trainFinal.append(maxm)
    
    #For Test Prediction
            
    for i in tqdm(range(len(testPredicted))):
        highest,itemsWithHighFrequency=np.argmax(np.bincount(testPredicted[i])),[]
        frequency = Counter(testPredicted[i]) 
        value = frequency[highest]
        
        for element,occurence in frequency.items():
            if occurence==value:
                itemsWithHighFrequency.append(element)
        
        if len(itemsWithHighFrequency)==1:
            testFinal.append(itemsWithHighFrequency[0])
        else:
            
            maxm,maxmScore=itemsWithHighFrequency[0],testSumScore[i,itemsWithHighFrequency[0]]
            for l in range(1,len(itemsWithHighFrequency)):
                if testSumScore[i,itemsWithHighFrequency[l]] > maxmScore:
                    maxmScore = testSumScore[i,itemsWithHighFrequency[l]]
                    maxm = itemsWithHighFrequency[l]
                    
            testFinal.append(maxm)
        
    #For validate Prediction
            
    for i in tqdm(range(len(validatePredicted))):
        highest,itemsWithHighFrequency=np.argmax(np.bincount(validatePredicted[i])),[]
        frequency = Counter(validatePredicted[i]) 
        value = frequency[highest]
        
        for element,occurence in frequency.items():
            if occurence==value:
                itemsWithHighFrequency.append(element)
        
        if len(itemsWithHighFrequency)==1:
            validateFinal.append(itemsWithHighFrequency[0])
        else:
            
            maxm,maxmScore=itemsWithHighFrequency[0],validateSumScore[i,itemsWithHighFrequency[0]]
            for l in range(1,len(itemsWithHighFrequency)):
                if validateSumScore[i,itemsWithHighFrequency[l]] > maxmScore:
                    maxmScore = validateSumScore[i,itemsWithHighFrequency[l]]
                    maxm = itemsWithHighFrequency[l]
                    
            validateFinal.append(maxm)
        
#     print("Train Output through Multi Class svm is ", accuracy_score(trainClassOut,trainFinal)*100," %")
    print("Test Output through Multi Class svm is ", accuracy_score(testClassOut,testFinal)*100," %")
    print("Validate Output through Mulit Class svm is ", accuracy_score(validateClassOut,validateFinal)*100," %")
    
    return testFinal,validateFinal
    
testPredPackage,validatePredPackage = multiClassPrediction(1.0,0.05)
# Here we are returning testPredicted and validatePredicted to be used for confusion matrix


# __Train Output through Multi Class svm is  96.52  %__
# 
# __Test Output through Multi Class svm is  85.08  %__
# 
# __Validate Output through Mulit Class svm is  84.96000000000001  %__

# In[5]:


def SVMMultiClass():
    
    trainClass=train[:,:-1]/255
    validateClass=validate[:,:-1]/255
    testClass=test[:,:-1]/255
    trainClassOut=train[:,-1].reshape(-1,1)
    validateClassOut=validate[:,-1].reshape(-1,1)
    testClassOut=test[:,-1].reshape(-1,1) 
    
# Here, we have trained different models and stored it using pickle. We would simply load it further computation.

#     models=[]
    
#     classes = [(i,j) for p,(i,j) in enumerate((i,j) for i in range(10) for j in range(i+1,10))]
    
#     for (class1,class2) in tqdm(classes):        
#         models.append(skLearn(1.0,0.05,class1,class2,"rbf"))
    
#     with open('skLrnGaussian','wb') as f:
#         pickle.dump(models,f)
        
#     with open('skLrnGaussian','rb') as f:
#         models=pickle.load(f)
      
    trainPredLabel,testPredLabel,valiPredLabel=[],[],[]
    trainScore,testScore,valiScore=[],[],[]
    
    for model in tqdm(models):
#         trainPredLabel.append(model.predict(trainClass))
#         trainScore.append(model.decision_function(trainClass))
        testPredLabel.append([int(x) for x in model.predict(testClass)])
        testScore.append(model.decision_function(testClass))
        valiPredLabel.append([int(x) for x in model.predict(validateClass)])
        valiScore.append(model.decision_function(validateClass))
#         break
    
    
#     print("testPredLabel ka type",type(testPredLabel))
#     print(testPredLabel[0])
    
#     trainPredLabel,trainScore=np.array(trainPredLabel).T,np.array(trainScore).T
    testPredLabel,testScore=np.array(testPredLabel).T,np.array(testScore).T
    valiPredLabel,valiScore=np.array(valiPredLabel).T,np.array(valiScore).T
    
#     print("after taking transpose checking for testPredLabel")
#     print(testPredLabel[0])


#     trainSumScore=np.zeros((trainClass.shape[0],10))
    testSumScore=np.zeros((testClass.shape[0],10))
    validateSumScore=np.zeros((validateClass.shape[0],10))
    
#     for i in range(len(trainPredLabel)):
#         for j in range(len(trainPredLabel[i])):
#             trainSumScore[i,trainPredLabel[i,j]]+=abs(trainScore[i,j])
    
    
#     print("printing length of testPredLabel ",len(testPredLabel))
    
    for i in range(len(testPredLabel)):
        for j in range(len(testPredLabel[i])):
            testSumScore[i,testPredLabel[i,j]]+=abs(testScore[i,j])
    
    for i in range(len(valiPredLabel)):
        for j in range(len(valiPredLabel[i])):
            validateSumScore[i,valiPredLabel[i,j]]+=abs(valiScore[i,j])
            
    #yahan saare score store kar liye
    
#     print("printing trainPredicted for trainPredicted")
#     print(trainPredicted)
    
    trainFinal,validateFinal,testFinal=[],[],[]

#     for i in tqdm(range(len(trainPredLabel))):
        
#         highest,itemsWithHighFrequency=np.argmax(np.bincount(trainPredLabel[i])),[]
#         frequency = Counter(trainPredLabel[i])
#         value = frequency[highest]
#         for element,occurence in frequency.items():
#             if occurence==value:
#                 itemsWithHighFrequency.append(element)
        
#         if len(itemsWithHighFrequency)==1:
#             trainFinal.append(itemsWithHighFrequency[0])
#         else:
#             maxm,maxmScore=itemsWithHighFrequency[0],trainSumScore[i,itemsWithHighFrequency[0]]
#             for l in range(1,len(itemsWithHighFrequency)):
#                 if trainSumScore[i,itemsWithHighFrequency[l]] > maxmScore:
#                     maxmScore = trainSumScore[i,itemsWithHighFrequency[l]]
#                     maxm = itemsWithHighFrequency[l]
                    
#             trainFinal.append(maxm)
    
    #For Test Prediction
            
    for i in tqdm(range(len(testPredLabel))):
        highest,itemsWithHighFrequency=np.argmax(np.bincount(testPredLabel[i])),[]
        frequency = Counter(testPredLabel[i]) 
        value = frequency[highest]
        
        for element,occurence in frequency.items():
            if occurence==value:
                itemsWithHighFrequency.append(element)
        
        if len(itemsWithHighFrequency)==1:
            testFinal.append(itemsWithHighFrequency[0])
        else:
            
            maxm,maxmScore=itemsWithHighFrequency[0],testSumScore[i,itemsWithHighFrequency[0]]
            for l in range(1,len(itemsWithHighFrequency)):
                if testSumScore[i,itemsWithHighFrequency[l]] > maxmScore:
                    maxmScore = testSumScore[i,itemsWithHighFrequency[l]]
                    maxm = itemsWithHighFrequency[l]
                    
            testFinal.append(maxm)
        
    #For validate Prediction
            
    for i in tqdm(range(len(valiPredLabel))):
        highest,itemsWithHighFrequency=np.argmax(np.bincount(valiPredLabel[i])),[]
        frequency = Counter(valiPredLabel[i]) 
        value = frequency[highest]
        
        for element,occurence in frequency.items():
            if occurence==value:
                itemsWithHighFrequency.append(element)
        
        if len(itemsWithHighFrequency)==1:
            validateFinal.append(itemsWithHighFrequency[0])
        else:
            
            maxm,maxmScore=itemsWithHighFrequency[0],validateSumScore[i,itemsWithHighFrequency[0]]
            for l in range(1,len(itemsWithHighFrequency)):
                if validateSumScore[i,itemsWithHighFrequency[l]] > maxmScore:
                    maxmScore = validateSumScore[i,itemsWithHighFrequency[l]]
                    maxm = itemsWithHighFrequency[l]
                    
            validateFinal.append(maxm)
        
#     print("Train Output through Multi Class svm is ", accuracy_score(trainClassOut,trainFinal)*100," %")
    print("Test Output through Multi Class svm is ", accuracy_score(testClassOut,testFinal)*100," %")
    print("Validate Output through Mulit Class svm is ", accuracy_score(validateClassOut,validateFinal)*100," %")
    
    return testFinal,validateFinal

testPredSklearn,validatePredSklearn=SVMMultiClass()
# Here we are returning testPredicted and validatePredicted to be used for confusion matrix


# __Test Output through Multi Class Svm is 88.08 %__
# 
# __Validate Output through Mulit Class svm is  87.88  %__

# <font size="3.5">__Q2 Part C Confusion Matrix__</font>

# In[12]:


get_ipython().magic('matplotlib qt')
def confusionMatrix(predictionParameter,actualParameter):
    
    predictionParameter=np.array(predictionParameter)
    paraConfusionMatrix = np.zeros((10,10),dtype=int).tolist()


    for i in range(10):
        for j in range(10):
            if i==j:
                paraConfusionMatrix[i][j]=np.sum(np.logical_and(actualParameter==predictionParameter,actualParameter==i))
            else:
                paraConfusionMatrix[i][j]=np.sum(np.logical_and(actualParameter==j, predictionParameter==i))
     
    plt.figure(figsize= (7,5))
    
    sn.heatmap(paraConfusionMatrix,cmap='YlOrBr', annot=True,cbar=False,fmt='d')
    
#     plt.title("Confusion Matrix", fontsize = 25)
    plt.xlabel("Actual Class", fontsize = 20)
    plt.ylabel("Predicted Class", fontsize = 20)
    plt.show()


testActual=np.array([int(item) for item in test[:,-1]])
validateActual=np.array([int(item) for item in validate[:,-1]])

#This function takes two arguments one for test and validation and it will give confusion matrix.
#pass the values for all four plots:-testPredPackage,validatePredPackage,testPredSklearn,validatePredSklearn

confusionMatrix(validatePredSklearn,validateActual)


# <font size="3.5">__Q2 Part D K-Fold Cross Validation__</font>

# In[ ]:


#Parallelized this portion of assignment for better 
from sklearn.model_selection import StratifiedKFold, cross_val_score
from joblib import Parallel, delayed
def kFoldCrossVal():
    
    trainData,testData=train[:,:-1]/255 ,test[:,:-1]/255
    trainOut,testOut=train[:,-1],test[:,-1]
    
    kfold = StratifiedKFold(n_splits=5)
    
#     value of c to be taken as 
    cList=[ 1e-5, 0.001, 1, 5, 10]
    outputList=[]
    
    for c in tqdm(cList):
        result=[]
        model=SVC(C=c,gamma=0.05,kernel="rbf")
        result.append(cross_val_score(model, trainData, trainOut, cv=kfold, n_jobs=-1))
        outputList.append(sum(result)/len(result))
    return  outputList
        
# out = Parallel(n_jobs=5, verbose=10)(delayed(kFoldCrossVal)(c) for c in tqdm([1e-5, 1e-3,1,5,10]))    

crossValidationAccuracy = kFoldCrossVal()
# crossValidationAccuracy,testAccuracy=zip(*out)
print(crossValidationAccuracy)

with open('crossValidateAccuracy','wb') as f:
        pickle.dump(crossValidationAccuracy,f)
        
# with open('kFoldTestAccuracy','wb') as f:
#     pickle.dump(testAccuracy,f)


# In[ ]:


#Now this is for the test Set Accuracies
from joblib import Parallel, delayed
def testAccuSkLearn(c):
  
  trainData = train[:,:-1]/255
  testData = test[:,:-1]/255
  trainOut = train[:,-1]
  testOut = test[:,-1]
  modelTest = SVC(C=c,gamma=0.05,kernel="rbf",decision_function_shape='ovo')
  modelTest.fit(trainData,trainOut.ravel())
  out = modelTest.score(testData,testOut)
  print(out)
  return out

testAccuracy = Parallel(n_jobs=5)(delayed(testAccuSkLearn)(c) for c in tqdm([1e-5, 1e-3,1,5,10]))
print(testAccuracy)


# __For different values of c these are test accuracies we got __
# 
# __[0.5736, 0.5736, 0.8808, 0.8828, 0.8824]__
# 
# __For different values of c these are validation accuracies we got__
# 
# __[0.5664444444444444, 0.5664444444444444, 0.8787111111111111, 0.8844, 0.8842666666666666]__

# In[97]:


def plotting():
    #Plotting For cross fold validation and test set accuracies
    get_ipython().magic('matplotlib qt')
    crossValidateAccuracy = (56.65, 56.65, 87.87, 88.44, 88.43)
    testAccuracy =          (57.36, 57.36, 88.08, 88.28, 88.24)
    
    
    fig,ax=plt.subplots()
    index=np.arange(5)
    width=0.12
    
    axis1 = np.array([math.log(1e-5,10), math.log(1e-3,10), math.log(1,10), math.log(5,10), math.log(10,10)])
    axis2 = axis1 + width
    
    print(axis1)
    print(axis2)
    
    plt1 = plt.bar(axis1,crossValidateAccuracy, width, color='y',label='Cross Validation Accuracy')
    plt2 = plt.bar(axis2, testAccuracy, width, color='b',label='Test Set Accuracy')
    
    plt.xlabel("Values of C")
    plt.ylabel("Accuracy in percentage")
    
#     plt.xticks(index+width,('1e-5', '1e-3', '1', '5', '10'))
    
    plt.xticks(axis1)
    
#     ax.set_xticks(index+width)
    
    ax.set_xlabel("Values of log C")
    ax.set_ylabel("Accuracy in percentage")
    ax.legend()
    plt.show()
    plt.legend()
    plt.tight_layout()
    plt.show()

plotting()

