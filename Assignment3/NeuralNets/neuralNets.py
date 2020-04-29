import pandas as pd
import time
import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# It takes two arguments from command line which is path for training and testing data

if len (sys.argv) != 3 :
    print("Please pass the required 2 arguments. ")
    sys.exit (1)

trainPath = sys.argv[1]
testPath = sys.argv[2]

train = pd.read_csv(trainPath,header = None).to_numpy()
test = pd.read_csv(testPath,header = None).to_numpy()

class neuralNets:

    def __init__(self,hiddenLayers,totalOutputClass,noOfFeatures,bSize,eta,activationType,convergenceCriteria,learnType):
        self.hiddenLayers = hiddenLayers
        self.totalOutputClass = totalOutputClass
        self.noOfFeatures = noOfFeatures
        self.bSize = bSize
        self.deltaJ = [i for i in range(len(hiddenLayers)+1)]
        self.outputs = [i for i in range(len(hiddenLayers)+1)]
        self.totalHiddenLayers = len(hiddenLayers)
        self.totalLayers = len(hiddenLayers)+1
        self.eta = eta
        self.epoch = 0
        self.learnType = learnType
        self.activationType = activationType
        self.convergenceCriteria = convergenceCriteria

    def initializeWeights(self,xInput):
        theta = []
        if self.totalHiddenLayers==0:
            totalThetas = self.totalOutputClass * (self.noOfFeatures+1)
            theta.append(np.random.uniform(-0.1,0.1,totalThetas).reshape(self.totalOutputClass,self.noOfFeatures+1))
        else:
            theta.append(np.random.uniform(-0.1,0.1,self.hiddenLayers[0]*(self.noOfFeatures+1)).reshape(self.hiddenLayers[0],self.noOfFeatures+1))
            for i in range(1,self.totalHiddenLayers):
                thetaSize = self.hiddenLayers[i] * (self.hiddenLayers[i-1]+1)
                theta.append(np.random.uniform(-0.1,0.1,thetaSize).reshape(self.hiddenLayers[i],(self.hiddenLayers[i-1]+1)))

            totalThetas = self.totalOutputClass * (self.hiddenLayers[-1]+1)
            theta.append(np.random.uniform(-0.1,0.1,totalThetas).reshape(self.totalOutputClass,(self.hiddenLayers[-1]+1)))

        self.theta = theta

    def costFun(self,predictedY,originalY): #CostFunction or LossFunction
        return (np.sum((originalY-predictedY)**2)/(2*originalY.shape[0]))

    def reluDerivate(self,xArray):
        return np.heaviside(xArray,0)

    def activation(self,xArray):
        if self.activationType == "sigmoid":
            return (1/(1+np.exp(-xArray)))
        if self.activationType == "relu":
            return np.maximum(0,xArray)

    def oneHotEncoder(self,yInput):
        self.oneHotEncodedY =  np.eye(self.totalOutputClass)[yInput]

    def feedForward(self,xInput,batchSize):
        intercept =  np.ones((batchSize,1))
        xFinally = np.append(intercept,xInput,axis=1)

        for j in range(self.totalHiddenLayers):
            dotProd = np.dot(xFinally,self.theta[j].T)
            out = self.activation(dotProd)
            self.outputs[j] = out
            xFinally = np.append(intercept,out,axis=1)

        outputLayerDot = np.dot(xFinally,self.theta[self.totalHiddenLayers].T)
        self.outputs[self.totalHiddenLayers] = (1/(1+np.exp(-outputLayerDot)))
        # We have seperated output layer as it's activation has to be sigmoid irrespective of activation type of hidden layers

    def backPropagation(self,xInputs,yInput):
        #For output layer
        yJ, oJ = yInput, self.outputs[self.totalLayers-1]
        yJMinusoJ, oneMinusoJ = (yJ - oJ) , (1 - oJ)
        oJMultiPly = oJ * oneMinusoJ
        delJ = oJMultiPly * yJMinusoJ
        self.deltaJ[-1] = delJ

        #Now for hidden layers
        for j in range(self.totalLayers-2,-1,-1):
            if self.activationType == "sigmoid":
                oJHidden = self.outputs[j]
                oJMultiPly = oJHidden * (1-oJHidden)
            if self.activationType == "relu":
                oJMultiPly = self.reluDerivate(self.outputs[j]) #It would calculate derivative of output with netJ

            delJThetaLJProd = np.dot(self.deltaJ[j+1],self.theta[j+1][:,1:])
            finalDelta = oJMultiPly * delJThetaLJProd
            self.deltaJ[j] = finalDelta

    def updateWeights(self,xInputs):
        for j in range(self.totalLayers):
            if j != 0:
                xPara = self.outputs[j-1]
            else:
                xPara = xInputs

            xWithIntercept = np.append(np.ones((self.bSize,1)),xPara,axis=1)
            delThetaJTheta = np.dot(self.deltaJ[j].T,xWithIntercept)/self.bSize
            self.theta[j] = self.theta[j] + (self.eta * delThetaJTheta)

    def fit(self,trainX,trainY):
        self.oneHotEncoder(trainY)
        self.initializeWeights(trainX)
        trainData = trainX[:,:self.noOfFeatures]
        totalInputs, minmIter = trainData.shape[0], int(trainData.shape[0]/self.bSize)
#         print(minmIter)

    #Now we will implement neural Nets using Stochastic Gradient Descent
        costInit, costFinal, i ,count = 100000, -100000, 0, 0

        while True:
#             if self.epoch == 2000:
#                 break

#             print(abs(costInit-costFinal))
            if abs(costInit-costFinal) < self.convergenceCriteria:
                break

            costPara = 0
            for l in range(minmIter):

                self.feedForward(trainData[i:i+self.bSize,:],self.bSize)
                costPara += self.costFun(self.outputs[-1],self.oneHotEncodedY[i:i+self.bSize,:])
                self.backPropagation(trainData[i:i+self.bSize,:],self.oneHotEncodedY[i:i+self.bSize,:])
                self.updateWeights(trainData[i:i+self.bSize,:])

                count+=1
                if (count*self.bSize) % totalInputs == 0:
                    self.epoch+=1
                    if self.learnType == "adaptive":  # For Part C
                        self.eta = 0.5/np.sqrt(self.epoch)

#                     print(self.epoch," ",abs(costInit-costFinal))

                i=(i+self.bSize)%totalInputs

            costInit = costFinal
            costFinal = costPara/minmIter

    def predict(self,dataInput):
        inputSize = dataInput.shape[0]
        self.feedForward(dataInput,inputSize)
        predictedOutput = self.outputs[-1]
        finalPredicted = np.array([np.argmax(predictedOutput[i]) for i in range(predictedOutput.shape[0])])
        return finalPredicted

def accuracy_score(originalY,predictedY):
    return np.sum(originalY==predictedY)/predictedY.shape[0]

def plottingAccuracy(x,yTrain,yTest):
    plt.plot(x,yTrain,label='Train Accuracy')
    plt.plot(x,yTest,label = 'Test Accuracy')
    plt.xlabel('Number of Hidden Layer Units')
    plt.ylabel('Accuracy in (%)')
    plt.legend()
    return plt

def plottingTime(x,yTime):
    plt.plot(x,yTime,label='Time taken to train')
    plt.xlabel('Number of Hidden Layer Units')
    plt.ylabel('Time taken to train the network (in secs)')
    plt.legend()
    return plt


# # Part B

hiddenLayerList = [1,5,50,100]

trainAccuracies, testAccuracies, timeToTrain, epochList = [], [], [], []

for units in hiddenLayerList:
    nn = neuralNets([units],26,784,100,0.1,"sigmoid",(1e-5),"normal")
    start = time.time()
    nn.fit(train[:,:-1]/255,train[:,-1])
    timeToTrain.append(time.time()-start)
    epochList.append(nn.epoch)
    predictedTrain, predictedTest = nn.predict(train[:,:-1]/255), nn.predict(test[:,:-1]/255)
    trainAccuracies.append((accuracy_score(train[:,-1],predictedTrain)*100))
    testAccuracies.append((accuracy_score(test[:,-1],predictedTest)*100))

print("Train accuracy for different hidden layer units are ")
print(trainAccuracies)
print("Test accuracy for different hidden layer units are ")
print(testAccuracies)
print("Time taken to train models for different hidden layer units are ")
print(timeToTrain)
print("Total number of epochs for different hidden layer units are ")
print(epochList)


# # Plotting for Part B

fig1 = plottingAccuracy(hiddenLayerList,trainAccuracies,testAccuracies)
fig1.show()

fig2 = plottingTime(hiddenLayerList,timeToTrain)
fig2.show()

# # Part C

trainAccuraciesAdap, testAccuraciesAdap, timeToTrainAdap, epochListAdap = [], [], [], []

for units in hiddenLayerList:
    nnAdap = neuralNets([units],26,784,100,0.1,"sigmoid",(1e-5),"adaptive")
    start = time.time()
    nnAdap.fit(train[:,:-1]/255,train[:,-1])
    timeToTrainAdap.append(time.time()-start)
    epochListAdap.append(nnAdap.epoch)
    predictedTrain, predictedTest = nnAdap.predict(train[:,:-1]/255), nnAdap.predict(test[:,:-1]/255)
    trainAccuraciesAdap.append((accuracy_score(train[:,-1],predictedTrain)*100))
    testAccuraciesAdap.append((accuracy_score(test[:,-1],predictedTest)*100))

print("Train accuracy for different hidden layer units for adaptive learning are ")
print(trainAccuraciesAdap)
print("Test accuracy for different hidden layer units for adaptive learning are ")
print(testAccuraciesAdap)
print("Time taken to train models for different hidden layer units for adaptive learning are ")
print(timeToTrainAdap)
print("Total number of epochs for different hidden layer units for adaptive learning are ")
print(epochListAdap)


# # Plotting for Part C

fig3 = plottingAccuracy(hiddenLayerList,trainAccuraciesAdap,testAccuraciesAdap)
fig3.show()

fig4 = plottingTime(hiddenLayerList,timeToTrainAdap)
fig4.show()

# # Part D

reluNet = neuralNets([100,100],26,784,100,0.1,"relu",(1e-5),"adaptive")
startRelu = time.time()
reluNet.fit(train[:,:-1]/255,train[:,-1])
print("Time taken to train Relu Network is ",time.time()-startRelu)
reluTrnPrd, reluTstPrd = reluNet.predict(train[:,:-1]/255), reluNet.predict(test[:,:-1]/255)
reluAcc = [accuracy_score(train[:,-1],reluTrnPrd),accuracy_score(test[:,-1],reluTstPrd)]
print("Train and test accuracy calculated by this network is ")
print(reluAcc)
print("Total number of epoch it took to train the Relu model is ",reluNet.epoch)

# # Part E

yToSend = np.eye(26)[train[:,-1]]
print(yToSend.shape)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

relumodel = MLPClassifier(hidden_layer_sizes=(100,100,),activation='relu',batch_size=100,solver='sgd',learning_rate='invscaling',learning_rate_init=0.5,max_iter=2000,momentum=0)
relst = time.time()
relumodel.fit(train[:,:-1]/255,yToSend)
print("Time it took to train the MLP relu model is ",time.time()-relst)

mlpReltrainProba = relumodel.predict_log_proba(train[:,:-1]/255)
mlpReltestProba = relumodel.predict_log_proba(test[:,:-1]/255)
mlpReltrainO = [np.argmax(item) for item in mlpReltrainProba]
mlpReltestO = [np.argmax(item) for item in mlpReltestProba]
print("MLP Relu Train Accuracy is ",accuracy_score(train[:,-1],mlpReltrainO))
print("MLP Relu Test Accuracy is ",accuracy_score(test[:,-1],mlpReltestO))
print("Total Iteration relu Model took is ",relumodel.n_iter_)
