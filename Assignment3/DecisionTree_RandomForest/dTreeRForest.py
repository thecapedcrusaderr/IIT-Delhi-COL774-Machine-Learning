#!/usr/bin/env python
# coding: utf-8

# In[1]:


from xclib.data import data_utils
import pandas as pd
import os
import sys
import numpy as np
import math
import statistics
import time
import pickle
from tqdm import tqdm
#Loading the data first

if len (sys.argv) != 7 :
    print("Please pass the required 6 arguments. ")
    sys.exit (1)

trnxPath = sys.argv[1]
trnyPath = sys.argv[2]
tstxPath = sys.argv[3]
tstyPath = sys.argv[4]
valxPath = sys.argv[5]
valyPath = sys.argv[6]

trainX = data_utils.read_sparse_file(trnxPath).toarray()
trainY = pd.read_csv(trnyPath,header=None).to_numpy()
testX = data_utils.read_sparse_file(tstxPath).toarray()
testY = pd.read_csv(tstyPath,header=None).to_numpy()
validX = data_utils.read_sparse_file(valxPath).toarray()
validY = pd.read_csv(valyPath,header=None).to_numpy()


# In[2]:


print(trainX.shape)
print(testX.shape)
print(validX.shape)


# # Tree with on the Go Accuracy

# In[3]:


class Node:
    def __init__(self):
        self.nodeCount = 0
        self.featureIndex = None # These all have been initialized to None
        self.isLeaf = True
        self.threshold = None #These are default values assigned which would change later.
        self.lChild = None
        self.rChild = None
        self.yLabel = None
        self.trainInd = None #Storing indexes of the data point coming to this node.
        self.testInd = None
        self.valiInd = None


# In[4]:


def findEntropy(yPara):
    if yPara.size == 0:
        return 0

    y0, y1 = yPara[yPara==0], yPara[yPara==1]
    lP, rP = y1.size/yPara.size, y0.size/yPara.size

    if lP == 0 or rP == 0:
        return 0
    else:
        return -(lP * math.log2(lP)) - (rP * math.log2(rP))


# In[5]:


def info_gain( index,thres, x, y):

    entropy = findEntropy(y)

    if entropy == 0:
        return -1
    else:

        lData = x[x[:,index] <= thres]
        rData = x[x[:,index] > thres]

        ly = y[ x[:,index] <= thres]
        ry = y[ x[:,index] > thres]


        #Returning -1 when data is either left skewed or right skewed

        if ly.size==0 or ry.size==0:
            return -1

        leftEntropy, rightEntropy = findEntropy(ly), findEntropy(ry)


        leftProba, rightProba = (ly.size/y.size), (ry.size/ y.size)
        condEntropy = leftProba * leftEntropy + rightProba * rightEntropy
        infoGain = entropy - condEntropy
        return infoGain


# In[6]:


def findMedians(xPara):
    if xPara.shape[0]%2 == 0:
        dataToAppend = np.zeros(xPara.shape[1]).reshape(1,-1)
        xPara = np.append(xPara,dataToAppend,axis=0)
        med = np.median(xPara,axis =0)
        return med
    else:
        med = np.median(xPara,axis=0)
        return med

#We already have trainX, trainY, testX, testY, validX, validY
#This function will handle everything through indexes only.
nodeC = 0 #It is global to handle node number of the nodes of the tree

trainPredict, testPredict, validPredict = np.copy(trainY), np.copy(testY), np.copy(validY)
trainAccuracyDict, testAccuracyDict, validAccuracyDict =  {},{},{}

def decisionTreeWithGo(nodePara,trainIndex,testIndex,valiIndex):
    global nodeC,trainPredict,testPredict,validPredict
    global trainAccuracyDict, testAccuracyDict, validAccuracyDict

    trainXPara = trainX[trainIndex]
    trainYPara = trainY[trainIndex]

    medians = findMedians(trainXPara.copy())

    indexOfMedians = np.array(list(range(medians.size)))

    mutualInfo,index,thresholdPara, = -sys.maxsize, -1, -1

    for(indexP,medP) in tqdm(zip(indexOfMedians,medians)):

        infoGain = info_gain(indexP, medP, trainXPara, trainYPara)

        if infoGain != -1 and infoGain > mutualInfo:
            mutualInfo = infoGain
            index = indexP
            thresholdPara = medP

    #Now we'll update trainY, testY and validY

    nodePara.trainInd = trainIndex
    nodePara.testInd = testIndex
    nodePara.valiInd = valiIndex

    nodePara.yLabel = np.bincount(trainYPara.flatten()).argmax()
    nodeC += 1
    nodePara.nodeCount = nodeC

    if testIndex.size != 0:
        testPredict[testIndex] = nodePara.yLabel
        testAccuracy = np.sum(testPredict==testY)/testY.size * 100
        testAccuracyDict[nodePara.nodeCount] = testAccuracy
    else:
        testAccuracyDict[nodePara.nodeCount] = testAccuracyDict[nodePara.nodeCount - 1]

    if valiIndex.size != 0:
        validPredict[valiIndex] = nodePara.yLabel
        validAccuracy = np.sum(validPredict==validY)/validY.size * 100
        validAccuracyDict[nodePara.nodeCount] = validAccuracy
    else:
        validAccuracyDict[nodePara.nodeCount] = validAccuracyDict[nodePara.nodeCount - 1]

    # Now for the train one, if it is leaf it won't enter the else part and the this function call execution
    # will be over.

#     print("printing trainIndex here ")
#     print(trainIndex)
#     print("nodePara.yLabel is ",nodePara.yLabel)
    trainPredict[trainIndex]=nodePara.yLabel
    trainAccuracy = np.sum(trainPredict==trainY)/trainY.size * 100
    trainAccuracyDict[nodePara.nodeCount] = trainAccuracy
#     print("printing accuracy for debugging")
#     print(trainAccuracyDict[nodePara.nodeCount])

    if index == -1:
        z = 2 # Dummy basically doing nothing
    else:
        nodePara.threshold = thresholdPara
        nodePara.featureIndex = index
        nodePara.isLeaf = False

        #Now we'll calculate indexes for left and right subTree of the given Node

        left, right = Node(), Node()

        trainLeftIndex = np.intersect1d((np.array(trainIndex),) , np.where(trainX[:,index] <= thresholdPara))
        trainRightIndex = np.intersect1d((np.array(trainIndex),) , np.where(trainX[:,index] > thresholdPara))

        testLeftIndex = np.intersect1d((np.array(testIndex),) , np.where(testX[:,index] <= thresholdPara))
        testRightIndex = np.intersect1d((np.array(testIndex),) , np.where(testX[:,index] > thresholdPara))

        validLeftIndex = np.intersect1d((np.array(valiIndex),) , np.where(validX[:,index] <= thresholdPara))
        validRightIndex = np.intersect1d((np.array(valiIndex),) , np.where(validX[:,index] > thresholdPara))

        yLeft = trainY[trainLeftIndex]
        yRight = trainY[trainRightIndex]

        lEntropy = findEntropy(yLeft)
        rEntropy = findEntropy(yRight)

        left.trainInd,left.testInd, left.valiInd = trainLeftIndex, testLeftIndex, validLeftIndex
        right.trainInd, right.testInd, right.valiInd = trainRightIndex, testRightIndex, validRightIndex

        nodePara.lChild, nodePara.rChild  = left, right

        if lEntropy==0 and rEntropy==0:
            nodePara.lChild.yLabel = np.bincount(yLeft.flatten()).argmax()
            nodeC+=1
            nodePara.lChild.nodeCount = nodeC

            #For Train Accuracy
            #It has come here, it means data will surely be partitioned, it won't be skewed.

            trainPredict[trainLeftIndex] = nodePara.lChild.yLabel
            trainAccuracy = np.sum(trainPredict==trainY)/trainY.size * 100
            trainAccuracyDict[nodePara.lChild.nodeCount] = trainAccuracy

            #For Test Accuracy
            if testLeftIndex.size == 0:
                testAccuracyDict[nodePara.lChild.nodeCount] = testAccuracyDict[nodePara.lChild.nodeCount - 1]
            else:
                testPredict[testLeftIndex]=nodePara.lChild.yLabel
                testAccuracy = np.sum(testPredict==testY)/testY.size * 100
                testAccuracyDict[nodePara.lChild.nodeCount] = testAccuracy

            #For Validate Accuracy
            if validLeftIndex.size == 0:
                validAccuracyDict[nodePara.lChild.nodeCount] = validAccuracyDict[nodePara.lChild.nodeCount -1]
            else:
                validPredict[validLeftIndex]=nodePara.lChild.yLabel
                validAccuracy = np.sum(validPredict==validY)/validY.size * 100
                validAccuracyDict[nodePara.lChild.nodeCount] = validAccuracy

            #******************Now for the  right subTree part*************************************

            nodePara.rChild.yLabel = np.bincount(yRight.flatten()).argmax()
            nodeC+=1
            nodePara.rChild.nodeCount = nodeC

            #For Train Accuracy
            #It has come here, it means data will surely be partitioned, it won't be skewed.

            trainPredict[trainRightIndex] = nodePara.rChild.yLabel
            trainAccuracy = np.sum(trainPredict==trainY)/trainY.size * 100
            trainAccuracyDict[nodePara.rChild.nodeCount] = trainAccuracy

            #For Test Accuracy
            if testRightIndex.size == 0:
                testAccuracyDict[nodePara.rChild.nodeCount] = testAccuracyDict[nodePara.rChild.nodeCount - 1]
            else:
                testPredict[testRightIndex]=nodePara.rChild.yLabel
                testAccuracy = np.sum(testPredict==testY)/testY.size * 100
                testAccuracyDict[nodePara.rChild.nodeCount] = testAccuracy

            #For Validate Accuracy
            if validRightIndex.size == 0:
                validAccuracyDict[nodePara.rChild.nodeCount] = validAccuracyDict[nodePara.rChild.nodeCount -1]
            else:
                validPredict[validRightIndex]=nodePara.rChild.yLabel
                validAccuracy = np.sum(validPredict==validY)/validY.size * 100
                validAccuracyDict[nodePara.rChild.nodeCount] = validAccuracy

        elif lEntropy==0:
            nodePara.lChild.yLabel = np.bincount(yLeft.flatten()).argmax()
            nodeC+=1
            nodePara.lChild.nodeCount = nodeC

            #For Train Accuracy
            #It has come here, it means data will surely be partitioned, it won't be skewed.

            trainPredict[trainLeftIndex] = nodePara.lChild.yLabel
            trainAccuracy = np.sum(trainPredict==trainY)/trainY.size * 100
            trainAccuracyDict[nodePara.lChild.nodeCount] = trainAccuracy

            #For Test Accuracy
            if testLeftIndex.size == 0:
                testAccuracyDict[nodePara.lChild.nodeCount] = testAccuracyDict[nodePara.lChild.nodeCount - 1]
            else:
                testPredict[testLeftIndex]=nodePara.lChild.yLabel
                testAccuracy = np.sum(testPredict==testY)/testY.size * 100
                testAccuracyDict[nodePara.lChild.nodeCount] = testAccuracy

            #For Validate Accuracy
            if validLeftIndex.size == 0:
                validAccuracyDict[nodePara.lChild.nodeCount] = validAccuracyDict[nodePara.lChild.nodeCount -1]
            else:
                validPredict[validLeftIndex]=nodePara.lChild.yLabel
                validAccuracy = np.sum(validPredict==validY)/validY.size * 100
                validAccuracyDict[nodePara.lChild.nodeCount] = validAccuracy

            #***************Now calling the function recursively for right subtree part*************

            decisionTreeWithGo(right,trainRightIndex,testRightIndex,validRightIndex)

        elif rEntropy==0:
            nodePara.rChild.yLabel = np.bincount(yRight.flatten()).argmax()
            nodeC+=1
            nodePara.rChild.nodeCount = nodeC

            #For Train Accuracy
            #It has come here, it means data will surely be partitioned, it won't be skewed.

            trainPredict[trainRightIndex] = nodePara.rChild.yLabel
            trainAccuracy = np.sum(trainPredict==trainY)/trainY.size * 100
            trainAccuracyDict[nodePara.rChild.nodeCount] = trainAccuracy

            #For Test Accuracy
            if testRightIndex.size == 0:
                testAccuracyDict[nodePara.rChild.nodeCount] = testAccuracyDict[nodePara.rChild.nodeCount - 1]
            else:
                testPredict[testRightIndex]=nodePara.rChild.yLabel
                testAccuracy = np.sum(testPredict==testY)/testY.size * 100
                testAccuracyDict[nodePara.rChild.nodeCount] = testAccuracy

            #For Validate Accuracy
            if validRightIndex.size == 0:
                validAccuracyDict[nodePara.rChild.nodeCount] = validAccuracyDict[nodePara.rChild.nodeCount -1]
            else:
                validPredict[validRightIndex]=nodePara.rChild.yLabel
                validAccuracy = np.sum(validPredict==validY)/validY.size * 100
                validAccuracyDict[nodePara.rChild.nodeCount] = validAccuracy

            #********Now calling the function recursively for left subTree
            decisionTreeWithGo(left,trainLeftIndex,testLeftIndex,validLeftIndex)
        else:
            #Now here we would call the function recursively for both left and right subtree
            decisionTreeWithGo(left,trainLeftIndex,testLeftIndex,validLeftIndex)
            decisionTreeWithGo(right,trainRightIndex,testRightIndex,validRightIndex)


# In[7]:


trainIndex = np.array(list(range(trainX.shape[0])))
testIndex = np.array(list(range(testX.shape[0])))
validIndex = np.array(list(range(validX.shape[0])))

root = Node()
start = time.time()
decisionTreeWithGo(root,trainIndex,testIndex,validIndex)
print("Time it took to build the decision tree is ",time.time()-start)


# In[8]:


import matplotlib.pyplot as plt
#Now plotting the graph
xValue = list(trainAccuracyDict.keys())
trainYValue = list(trainAccuracyDict.values())
testYValue = list(testAccuracyDict.values())
validYValue = list(validAccuracyDict.values())

def plottingOnTheGo(xValue,trainYValue,testYValue,validYValue):
    plt.plot(xValue,trainYValue,label='Train Accuracy')
    plt.plot(xValue,testYValue,label = 'Test Accuracy')
    plt.plot(xValue,validYValue,label = 'Validation Accuracy')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Accuracies in (%)')
    plt.legend()
    plt.show()

plottingOnTheGo(xValue,trainYValue,testYValue,validYValue)


# # Post Pruning of Decision Tree

# In[ ]:


#sabse pehle root pickle krunga
# import pickle
# pickle.dump(root,open('savedRoot','wb'))


# In[ ]:


#just storing these predicted values to reduce further efforts
# trainSave = trainPredict.copy()
# testSave = testPredict.copy()
# valiSave = validPredict.copy()


# In[ ]:


# pickle.dump(trainSave,open('trainPredict','wb'))
# pickle.dump(testSave,open('testPredict','wb'))
# pickle.dump(valiSave,open('validPredict','wb'))


# In[ ]:


# trainPredict = pickle.load(open('trainPredict','rb'))
# testPredict = pickle.load(open('testPredict','rb'))
# validPredict = pickle.load(open('validPredict','rb'))


# In[ ]:


# root = pickle.load(open('savedRoot','rb'))


# In[9]:


# print(trainPredict.size)
# print(testPredict.size)
# print(validP)

#just checking
print("Train Accuracy is ",np.sum(trainPredict==trainY)/trainY.size * 100)
print("Test Accuracy is ",np.sum(testPredict==testY)/testY.size * 100)
print("Validation Accuracy is ",np.sum(validPredict==validY)/validY.size * 100)


# In[10]:


def findNodeCount(node):
    if node.isLeaf==True:
        return 1
    else:
        return 1 + findNodeCount(node.lChild) + findNodeCount(node.rChild)


# # Post Pruning using Iterative approach

# #Will Mark down it as we will use bottom up approach
# #Iterative way of post pruning
# #We have got root as our rootNode of fully grown tree.
#
# pruneTrainAcc, pruneTestAcc, pruneValAcc = {}, {}, {}
#
# #Finding the final node count after the tree have fully grown
#
# finalTotalNode = findNodeCount(root)
#
# pruneTrainAcc[finalTotalNode] = trainAccuracyDict[finalTotalNode]
# pruneTestAcc[finalTotalNode] = testAccuracyDict[finalTotalNode]
# pruneValAcc[finalTotalNode] = validAccuracyDict[finalTotalNode]
#
# #Maintaining global validation accuracy
# valAccuracy = pruneValAcc[finalTotalNode]
#
# def treePruning():
#
#     nodeToPrune = None
#
#     def postPruning(node):
#
#         global valAccuracy
#         nonlocal nodeToPrune
#
#         valPre = np.copy(validPredict)
#
#         if node.isLeaf == False: #and node.valiInd.size != 0:
#             valPre[node.valiInd] = node.yLabel
#             validAcc = np.sum(valPre==validY)/validY.size * 100
#             if validAcc >= valAccuracy:
#
#                 valAccuracy = validAcc
#                 nodeToPrune = node
#
#             postPruning(node.lChild)
#             postPruning(node.rChild)
#
#     postPruning(root)
#
#     if nodeToPrune != None:
#
#         trainPredict[nodeToPrune.trainInd] = nodeToPrune.yLabel
#         trainAcc = np.sum(trainPredict==trainY)/trainY.size * 100
#         testPredict[nodeToPrune.testInd] = nodeToPrune.yLabel
#         testAcc = np.sum(testPredict==testY)/testY.size * 100
#         validPredict[nodeToPrune.valiInd] = nodeToPrune.yLabel
#         validAcc = np.sum(validPredict==validY)/validY.size * 100
#
#         nodeToPrune.isLeaf = True #Only this would suffice as we are making it a leaf
#         nNode = findNodeCount(root)
#         pruneTrainAcc[nNode] = trainAcc
#         pruneTestAcc[nNode] = testAcc
#         pruneValAcc[nNode] = validAcc
#
#         #calling function recursively now
#         treePruning()
#
# treePruning()

# In[ ]:


# pickle.dump(pruneTrainAcc,open('iterPrunTrain','wb'))
# pickle.dump(pruneTestAcc,open('iterPrunTest','wb'))
# pickle.dump(pruneValAcc,open('iterPrunVal','wb'))


# In[ ]:


# pruneTrain = pickle.load(open('iterPrunTrain','rb'))
# pruneTest = pickle.load(open('iterPrunTest','rb'))
# pruneVal = pickle.load(open('iterPrunVal','rb'))


# # Post Pruning using Bottom Up (Post Order Traversal) Approach

# In[11]:


#Recursive approach for pruning

#We have trainPredict, testPredict and validPredict obtained after we have grown full tree.


#They will contain accuracy as we prune the tree
recurTrainAcc, recurTestAcc, recurValAcc = {},{},{}

finalTotalNode = findNodeCount(root)

recurTrainAcc[finalTotalNode] = trainAccuracyDict[finalTotalNode]
recurTestAcc[finalTotalNode] = testAccuracyDict[finalTotalNode]
recurValAcc[finalTotalNode] = validAccuracyDict[finalTotalNode]

currentValAcc = recurValAcc[finalTotalNode]

def recurPruning(node):

    global currentValAcc
    if node.isLeaf == False:

        recurPruning(node.lChild)
        recurPruning(node.rChild)

        val = validPredict.copy()
        val[node.valiInd] = node.yLabel
        prediction = np.sum(val==validY)/validY.size * 100

        if prediction >= currentValAcc:
            currentValAcc = prediction
            node.isLeaf = True
            validPredict[node.valiInd] = node.yLabel
            trainPredict[node.trainInd] = node.yLabel
            testPredict[node.testInd] = node.yLabel
            valP = np.sum(validPredict==validY)/validY.size * 100
            trnP = np.sum(trainPredict==trainY)/trainY.size * 100
            tstP = np.sum(testPredict==testY)/testY.size * 100
            nC = findNodeCount(root)
            recurTrainAcc[nC] = trnP
            recurTestAcc[nC] = tstP
            recurValAcc[nC] = valP


# In[12]:


recurPruning(root)


# In[ ]:


# print(findNodeCount(root))


# In[ ]:


# pickle.dump(recurTrainAcc,open('recurPruneTrain','wb'))
# pickle.dump(recurTestAcc,open('recurPruneTest','wb'))
# pickle.dump(recurValAcc,open('recurPruneVal','wb'))


# In[ ]:


# pruneTrain = pickle.load(open('recurPruneTrain','rb'))
# pruneTest = pickle.load(open('recurPruneTest','rb'))
# pruneVal = pickle.load(open('recurPruneVal','rb'))


# In[13]:


def prunePlotting(pruneX,pruneTrnAcc,pruneTstAcc,pruneValAcc):
    plt.plot(pruneX,pruneTrnAcc,label='Train Accuracy')
    plt.plot(pruneX,pruneTstAcc,label = 'Test Accuracy')
    plt.plot(pruneX,pruneValAcc,label = 'Validation Accuracy')
    ax=plt.gca()
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel('Number of Nodes')
    plt.ylabel('Accuracies in (%) after pruning')
    plt.legend()
    plt.show()

xAxisVal = list(recurTrainAcc.keys())
trainYVal = list(recurTrainAcc.values())
testYVal = list(recurTestAcc.values())
valYVal = list(recurValAcc.values())

prunePlotting(xAxisVal,trainYVal,testYVal,valYVal)


# # Random Forest

# In[ ]:


#Let's start the sklearn part here
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#We have to vary the paramter values in this range as given in the question.

n_est = list(range(50,451,100))
max_feat = [0.1,0.3,0.5,0.7,0.9]
min_samp_spl = list(range(2,11,2))

def scorer(modelPara,xPara,yPara):
  return modelPara.oob_score_

# score = scorer()

clf = RandomForestClassifier(n_jobs = -1, oob_score = True)
# st = time.time()
parameters = {'n_estimators':n_est ,'max_features':max_feat,'min_samples_split': min_samp_spl}
model = GridSearchCV(clf, parameters, cv=5, scoring = scorer ,verbose = 2)
model.fit(trainX,trainY.flatten())
# print("Time it took for model for fitting in is ",time.time()-st)

# with open('model','wb') as f:
#     pickle.dump(model,f)

best_params = model.best_params_
best_max_feat = best_params['max_features']
best_min_samp_split = best_params['min_samples_split']
best_n_estimators = best_params['n_estimators']

finalModel = RandomForestClassifier(n_estimators = best_n_estimators, min_samples_split = best_min_samp_split, max_features = best_max_feat,oob_score = True)
finalModel.fit(trainX,trainY.flatten())

print("Oob Score for training data for best learned parameters is ", finalModel.oob_score_ * 100," %")
print("Accuracy score for training data for best learned parameters is ",accuracy_score(trainY,finalModel.predict(trainX)) * 100," %")
print("Accuracy score for test data for best learned parameters is ",accuracy_score(testY,finalModel.predict(testX)) * 100," %")
print("Accuracy score for validation data for best learned parameters is ",accuracy_score(validY,finalModel.predict(validX)) * 100," %")


# __{'max_features': 0.1, 'min_samples_split': 10, 'n_estimators': 450}__

# __Oob Score for training data for best learned parameters are 81.07644522738863  %__
#
# __Accuracy score for training data for best learned parameters are  87.3642081189251  %__
#
# __Accuracy score for test data for best learned parameters are  80.82610912799592  %__
#
# __Accuracy score for validate data for best learned parameters are  80.73428518449842  %__
#

# # Random Forest - Parameter Sensitivity Analysis

# In[ ]:


#Now Part D of sklearn of the assignment
#{'max_features': 0.1, 'min_samples_split': 10, 'n_estimators': 450}

#From the above part we have got best paramters in the variables best_max_feat, best_min_samp_split, best_n_estimators

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
from itertools import product

#Varying the parameters in this range

n_est = list(range(50,1001,50))
max_feat = np.linspace(0.01,1.0,20)
min_samp_spl = list(range(2,60,3))

varyfeatModel = list(product(max_feat,[best_min_samp_split],[best_n_estimators]))
varyMinSampModel = list(product([best_max_feat],min_samp_spl,[best_n_estimators]))
varyNEstim = list(product([best_max_feat],[best_min_samp_split],n_est))

finalList = varyfeatModel+varyMinSampModel+varyNEstim

#Here training all 60 models

def varyParaAccuracy(max_feat,min_samp_split,n_estim):
  model = RandomForestClassifier(max_features = max_feat, min_samples_split=min_samp_split, n_estimators = n_estim)
  model.fit(trainX,trainY.flatten())
  return model

finalModels = Parallel(n_jobs = -1)(delayed(varyParaAccuracy)(x,y,z) for (x,y,z) in tqdm(finalList))


# In[ ]:


from sklearn.metrics import accuracy_score

#Slicing different models as we have done trained 60 models.

models_var_feat = finalModels[:20]
models_min_sampsplit = finalModels[20:40]
models_nEstimators = finalModels[40:]

validAccVaryFeat = np.array([accuracy_score(validY,valiModel.predict(validX)) for valiModel in models_var_feat]) * 100
validAccVaryMinSample = np.array([accuracy_score(validY,valiModel.predict(validX)) for valiModel in models_min_sampsplit]) * 100
validAccVaryNestimators = np.array([accuracy_score(validY,valiModel.predict(validX)) for valiModel in models_nEstimators]) * 100

testAccVaryFeat = np.array([accuracy_score(testY,testModel.predict(testX)) for testModel in models_var_feat]) * 100
testAccVaryMinSample = np.array([accuracy_score(testY,testModel.predict(testX)) for testModel in models_min_sampsplit]) * 100
testAccVaryNestimators = np.array([accuracy_score(testY,testModel.predict(testX)) for testModel in models_nEstimators]) * 100


# In[ ]:

import matplotlib.pyplot as plt
def plottingAccuracy(x,ty,vy,xlab,ylab):
    plt.plot(x,ty,label='Test Accuracy')
    plt.plot(x,vy,label='Validation Accuracy')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
#     plt.ylim(75,95)
    plt.legend()
    plt.show()


# In[ ]:


plottingAccuracy(max_feat,testAccVaryFeat,validAccVaryFeat,'max_features','Accuracy in (%) varying max_features')


# In[ ]:


plottingAccuracy(min_samp_spl,testAccVaryMinSample,validAccVaryMinSample,'min_samples_split','Accuracy in (%) varying min_samples_split')


# In[ ]:


plottingAccuracy(n_est,testAccVaryNestimators,validAccVaryNestimators,'n_estimators','Accuracy in (%) varying n_estimators')
