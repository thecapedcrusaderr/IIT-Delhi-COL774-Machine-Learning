
# coding: utf-8

# In[3]:


import numpy as np
import math
import time
import pickle
import random
import pandas as pd
import numpy as np
import seaborn as sn
import collections
from tqdm import tqdm
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter


# In[4]:


train=pd.read_csv("training.1600000.processed.noemoticon.csv",encoding='latin-1',header=None)
test=pd.read_csv("testdata.manual.2009.06.14.csv",encoding='latin-1',header=None)
testRef=np.array(test)
check=testRef[testRef[:,0]!=2]
test=pd.DataFrame(check)
# print(test)


# In[24]:


def mainFunction(dataAttr,ngram):
    
    def trainInitialization(dataAttr):  #0 for Negative and 4 for Positive
        vocabData=[]
        dataForVocab=np.array(dataAttr[5])
        classData=np.array(dataAttr[0])
        
        print("yahan aa bhi rha hai kya")
        
        adverb=["RB","RBR","RBS","WRB"]
        verb=["VB","VBD","VBG","VBN","VBP","VBZ"]
        adjective=["JJ","JJR","JJS"]
        
        tempIndex=0
        for sentence in tqdm(dataForVocab):
            temp=sentence.replace(","," ").replace("."," ").split()
            
            for words in temp:
                vocabData.append(words)
                
# Below is the code for POS tagging which would be uncommented when needed.
            
#             newSentence=finalTaggedList[tempIndex]
#             for (word,tagged) in newSentence:
#                 if "RB" in tagged:
#                     temp=[word]*3
#                     vocabData=vocabData+temp
#             tempIndex+=1
                
            if(ngram=="bi"):
                temp2 = sentence.replace(","," ").replace("."," ").split()
                for i in range(len(temp2)-1):
                    vocabData.append(temp2[i]+" "+temp2[i+1])
        
#         with open('vocabData','wb') as f:
#             #It is being stored for clean data only
#             pickle.dump(vocabData,f)
                

        trainNegative=dataAttr[dataAttr[0]==0]
        trainPositive=dataAttr[dataAttr[0]==4]

        vocabPositiveData,vocabNegativeData=[],[]
        trainPositiveTweet,trainNegativeTweet=trainPositive[5],trainNegative[5]
        
        posTweetIndex=0
        for sentence in tqdm(trainPositiveTweet):
            temp=sentence.replace(","," ").replace("."," ").split()
            for words in temp:
                vocabPositiveData.append(words)
  

#   Below is the code that would be uncommented when needed.
#               posSentence=positiveTaggedList[posTweetIndex]
#             for (word,tagged) in posSentence:
#                 if "RB" in tagged:
#                     temp=[word]*3
#                     vocabPositiveData=vocabPositiveData+temp
#             posTweetIndex+=1
            
            
            if(ngram=="bi"):
                temp2 = sentence.replace(","," ").replace("."," ").split()
                for i in range(len(temp2)-1):
                    vocabPositiveData.append(temp2[i]+" "+temp2[i+1])
        
#         with open('vocabPositiveData','wb') as f:
#             #It is being stored for clean data only, vocabPositiveData
#             pickle.dump(vocabPositiveData,f)

        negTweetIndex=0
        for sentence in tqdm(trainNegativeTweet):
            temp=sentence.replace(","," ").replace("."," ").split()
            for words in temp:
                vocabNegativeData.append(words)

# Below is the code for POS tagging which would be uncommented when needed.

#             negSentence=negativeTaggedList[negTweetIndex]
#             for (word,tagged) in negSentence:
#                 if "RB" in tagged:
#                     temp=[word]*3
#                     vocabNegativeData=vocabNegativeData+temp
#             negTweetIndex+=1  
            
                
            if(ngram=="bi"):
                temp2 = sentence.replace(","," ").replace("."," ").split()
                for i in range(len(temp2)-1):
                    vocabNegativeData.append(temp2[i]+" "+temp2[i+1])
                    
            
        
#         with open('vocabNegativeData','wb') as f:
#             #It is being stored for clean data only, vocabPositiveData
#             pickle.dump(vocabNegativeData,f)

        return vocabData,vocabPositiveData,vocabNegativeData,classData
    
    start=time.time()
    
    (vocabData,vocabPositiveData,vocabNegativeData,classData)=trainInitialization(dataAttr)
    
    def afterTrainInitialization():
        totalPositiveWord=len(vocabPositiveData)
        totalNegativeWord=len(vocabNegativeData)
        print("Total Negative Word : ",totalNegativeWord)
        print("---------------------")
        countPositiveWord=Counter(vocabPositiveData)
        countNegativeWord=Counter(vocabNegativeData)
        print(totalPositiveWord,totalNegativeWord,len(countPositiveWord),len(countNegativeWord))
        return totalPositiveWord,totalNegativeWord,countPositiveWord,countNegativeWord
    
    (totalPositiveWord,totalNegativeWord,countPositiveWord,countNegativeWord)= afterTrainInitialization()
    
    def phi():
        temp=[]
        classCount=Counter(classData)
        total=sum(classCount.values())
        temp.append((classCount[0]+1)/(total+2))
        temp.append((classCount[4]+1)/(total+2))
        return temp

    phii=phi()
    print(phii)
    
    def indexForWords():
        vocabWords=Counter(vocabData)
        words=list(vocabWords.keys())

        wordsIndex = collections.defaultdict(lambda : -1)
        p=0
        for word in words:
            wordsIndex[word]=p
            p+=1

        #Checking default dict in python

        sizeVocab=(len(words))
        print(sizeVocab)
        return wordsIndex,words,sizeVocab
    
    wordsIndex,words,sizeVocab=indexForWords()
    
    def theta():
        temp1,temp2=[],[]
        for x in words:
            temp1.append((countNegativeWord[x]+1)/(totalNegativeWord+sizeVocab))
            temp2.append((countPositiveWord[x]+1)/(totalPositiveWord+sizeVocab))

        temp3=[temp1,temp2]
        print(len(temp3))
        return temp3

    theta=theta()
    
    return theta,wordsIndex,phii,totalPositiveWord,totalNegativeWord,sizeVocab

start=time.time()
(theta,wordsIndex,phii,totalPositiveWord,totalNegativeWord,sizeVocab)=mainFunction(cleanTrainData.copy(),"uni")
print("Time taken for training is ",time.time()-start)

#The second parameter of mainFunction is for feature engineering that takes "bi" for including bigram with unigram
#Here we have to call to get theta for testing the generated model
# out=mainFunction(train)


# In[166]:


def testData(data,ngram):
    attrPanda=pd.DataFrame(data)
    tweets=attrPanda[5]
    actualValue=[]
    prob=[]
    k=0
    for sentence in tqdm(tweets):
        resPos,resNeg=0,0
        
        temp1=sentence.replace(","," ").replace("."," ").split()
        if(ngram=="bi"):
                temp2 = sentence.replace(","," ").replace("."," ").split()
                for i in range(len(temp2)-1):
                    temp1.append(temp2[i]+" "+temp2[i+1])
        
        k+=1
        
        for words in temp1:

            if(wordsIndex[words]!=-1):
                resPos+=math.log(theta[1][wordsIndex[words]])
                resNeg+=math.log(theta[0][wordsIndex[words]])
            else:
                resPos+=math.log(1/(totalPositiveWord+sizeVocab))
                resNeg+=math.log(1/(totalNegativeWord+sizeVocab))
        resPos+=math.log(phii[1])
        resNeg+=math.log(phii[0])
        prob.append(resPos)
        if(resPos>resNeg):
            actualValue.append(4)
        else:
            actualValue.append(0)
    originalValue=list(attrPanda[0])
    return originalValue,actualValue,prob

(original,tested,probability)=testData(test.copy(),"uni")


# In[169]:


get_ipython().magic('matplotlib qt')
def cMatrixAndAccuracy(original,tested):
    l=len(original)
    c00,c04,c40,c44=0,0,0,0
    for (o,t) in zip(original,tested):
        if o==0 and t==0:
            c00+=1
        elif o==0 and t==4:
            c04+=1
        elif o==4 and t==0:
            c40+=1
        else:
            c44+=1
    accuracy=((c00+c44)/l)*100
    data=[[c00,c40],[c04,c44]]
    print("printing",c00,c40,c04,c44)
#     matrix=pd.DataFrame(data,columns=['Class0','Class4'],index=['Class0', 'Class4'])
    return accuracy,data

accuracy,matrix=cMatrixAndAccuracy(original,tested)
print("Accuracy is ",accuracy)
     
plt.figure(figsize= (7,4))

sn.heatmap(matrix,cmap='OrRd', annot=True,cbar=False,fmt='d', xticklabels=[0,4], yticklabels=[0,4])

# plt.title("Confusion Matrix", fontsize = 10)
plt.xlabel("Actual Class", fontsize = 10)
plt.ylabel("Predicted Class", fontsize = 10)
plt.show()

# print("Confusion matrix is :","\n")
# print(matrix)
# print("Time taken is : ", time.time()-start)


# __Accuracy for test when unigram is used : 81.33704735376045%__
# 
# __Accuracy for test when bigram is used : 83.84401114206128 %__

# __Question 1:b__

# In[50]:


def differentAccuracies():
    testOriginal=list(test[0])
    lengthTest=len(testOriginal)
#     print(lengthTest)
    testRandom=[]

    for i in range(lengthTest):
        testRandom.append(random.choice([0,4]))
    
    testMajority0=[0]*lengthTest
    testMajority4=[4]*lengthTest

    #Accuracy for randomly assigned outputs
    accuRandom,matRandom=cMatrixAndAccuracy(testOriginal,testRandom)
    accuMajority0,matMajority0=cMatrixAndAccuracy(testOriginal,testMajority0)
    accuMajority4,matMajority4=cMatrixAndAccuracy(testOriginal,testMajority4)

    print("Accuracy in case of random generation is : ",accuRandom)
    print("Accuracy in case of Majority Predicton with 0 as majority is : ",accuMajority0)
    print("Accuracy in case of Majority Predicton with 4 as majority is : ",accuMajority4)
    
differentAccuracies()


# __Question 1:d__

# In[7]:


def removingStopWords(dataToStem):
    #Here dataToStem is to be a panda DataFrame
    
    ps = PorterStemmer()
    tweets=dataToStem[5]
    stop_words = set(stopwords.words('english'))
    finalSentenceList=[]
    i=0
    for sentences in tqdm(tweets):
        temp=sentences.replace(","," ").replace("."," ").split()
        
        afterTwitterHandle=[word for word in temp if not word.startswith('@')]
        
        afterStopWord=[word for word in afterTwitterHandle if word not in stop_words]
        
        finalList=[ps.stem(word) for word in afterStopWord]
        
        sen=' '.join(finalList)
        
        finalSentenceList.append(sen)

    justForTest=pd.DataFrame(finalSentenceList)
    dataToStem[5]=justForTest
    return dataToStem

# print(test[5])
cleanTestData=removingStopWords(test.copy())
# print(cleanTestData[5])
# print("checking if train is getting changed or not")
# print(train)
cleanTrainData=removingStopWords(train.copy())
# print("checking if train data is changed or not")
# print(train)
# print(cleanTrainData)
# print(cleanTestData)


# __Tested On Cleaned Trained Data__
# Accuracy is  81.0645
# Confusion matrix is : 
# 
#         Class0  Class4
# Class0  670501  129499
# Class4  173469  626531
# 
# 

# __Tested on Cleaned Tested Data__ 
# printing 144 33 34 148
# Accuracy is  81.33704735376045
# Confusion matrix is : 
# 
#         Class0  Class4
# Class0     144      33
# Class4      34     148

# __Tested on cleaned Test Data having Trained on original train data__
# Accuracy is  77.15877437325905
# Confusion matrix is : 
# 
#         Class0  Class4
# Class0     132      45
# Class4      37     145
# 

# __Tested on Original Test Data having trained on Original Trained Data__
# Accuracy is  80.77994428969359
# Confusion matrix is : 
# 
#         Class0  Class4
# Class0     142      35
# Class4      34     148
# 

# __Part 1(D) onwards__

# # TF-IDF Part A

# In[162]:


#Here one extra parameter min_df is added to increase the accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
def tf_idf():
    cleanedTrainTweet=list(cleanTrainData[5])
    cleanedTestTweet=list(cleanTestData[5])
    vectorizer = TfidfVectorizer() #min_df=0.0006 include it in parameter in vectorizer when need min-Df
    trainFeatures=vectorizer.fit_transform(cleanedTrainTweet)
    testFeatures=vectorizer.transform(cleanedTestTweet)
    return trainFeatures,testFeatures
trainFeatures,testFeatures=tf_idf() 


# In[ ]:


print(trainFeatures.shape)
print("----------------------------")
print(testFeatures.shape)


# In[6]:


#Learning on training data model has been generated and stored so it will be just used later
from sklearn.naive_bayes import GaussianNB
def generateModel(trainAttrFeatures):
    model = GaussianNB()
    expectedOutput=cleanTrainData[0].to_numpy()
    classes=np.unique(expectedOutput)
    for i in tqdm(range(0,cleanTrainData[0].size,1000)):
        model.partial_fit(trainAttrFeatures[i:i+1000].toarray(),expectedOutput[i:i+1000],classes)
    return model


# In[ ]:


#Model already there
# with open('model','wb') as f:
#     pickle.dump(model,f)


# In[144]:


# with open('model','rb') as f:
#     model=pickle.load(f)    


# In[110]:


def accuracy(attrModel,testAttrFeatures):
    actualOutput=attrModel.predict(testAttrFeatures[:].toarray())
    prob=attrModel.predict_log_proba(testAttrFeatures[:].toarray())[:,1]
    accu=attrModel.score(testAttrFeatures[:].toarray(),list(cleanTestData[0]))
    return accu,prob


# __For accuracy of model on all the features__

# In[ ]:


accuracyIncludingAll,probabilityIncludingAll=accuracy(model,testFeatures)
print(accuracyIncludingAll)


# __ Accuracy for all the features is 49.58217270194986__

# In[130]:


mindfModel=generateModel(trainFeatures)
accuracyForMinDf,probabilityForMinDf=accuracy(mindfModel,testFeatures)
print("Accuracy using minDf Model is ",accuracyForMinDf * 100," %")


# __Accuracy using MinDf for tfidf without using select percentile value is 78.55153203342619__

# # TF-IDF Part B

# In[163]:


from sklearn.feature_selection import SelectPercentile, f_classif
def tf_idfSelectPercentile():
    trainFeat,testFeat=tf_idf()
    support = SelectPercentile(f_classif, percentile=35)
    support.fit(trainFeat,list(cleanTrainData[0]))
    
    trainPerFeatures = support.transform(trainFeat)
    testPerFeatures = support.transform(testFeat)
    
    return trainPerFeatures,testPerFeatures
st=time.time()
trainPerFeatures,testPerFeatures=tf_idfSelectPercentile() 


# In[ ]:


percentileModel=generateModel(trainPerFeatures)
print("time taken is ",time.time()-st)


# In[ ]:


#Storing Percentile Model
# with open('percentileModel','wb') as f:
#     pickle.dump(percentileModel,f)


# In[151]:


#Accessing Percentile Model
# with open('percentileModel','rb') as f:
#     percentileModel=pickle.load(f)    


# In[ ]:


print(testPerFeatures.shape)


# In[127]:


afterPercentileAccuracy,percentileProbability=accuracy(percentileModel,testPerFeatures)
print(afterPercentileAccuracy)


# __Accuracy after Select 10 Percentile is 55.71030640668524__ 
# 
# __Accuracy after Select 10 Percentile using MinDf as 0.0006 is 71.3091922005571__ 
# 
# __Accuracy after Select 30 Percentile is 49.303621169916434__
# 
# __Accuracy after Select 5 Percentile is 57.93871866295265__
# 
# __Accuracy after Select 5 Percentile using MinDf as 0.0006 is 64.62395543175488__
# 
# __Accuracy after Select 2 Percentile is 69.63788300835655__
# 
# __Accuracy after Select 2 Percentile using MinDf as 0.0006 is 60.16713091922006__
# 
# __Accuracy after Select 35 Percentile using MinDf as 0.0006 is 77.71587743732591__

# # ROC CURVE

# In[134]:


get_ipython().magic('matplotlib qt')
from sklearn import metrics
def roc(originOut,prob):
    import matplotlib.pyplot as plt
    falsePositive,truePositive,thresholds = metrics.roc_curve(originOut, prob, pos_label=4)
    auc = metrics.roc_auc_score(originOut,prob)
    plt.plot(falsePositive,truePositive,label='Area under curve is '+str(auc))
    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.legend()
    plt.show()


# In[137]:


# Roc curve for test having trained on Original Train Data
# Always check for the parameter of main function, that's where training is being done.
(original,tested,probability)=testData(test.copy(),"uni")
roc(original,probability)


# In[141]:


#Roc curve for cleaned test data having trained on Cleaned Test Data
(original,tested,probability)=testData(cleanTestData.copy(),"uni")
roc(original,probability)


# In[145]:


#Roc curve obtained after tfidf with 100 percentile, these all will be done on cleaned data only
accuracyIncludingAll,probabilityIncludingAll=accuracy(model,testFeatures)
originalValue=list(cleanTestData[0])
roc(originalValue,probabilityIncludingAll)


# In[152]:


#Roc curve obtained after tfidf with 10 percentile,for cleaned data only
afterPercentileAccuracy,percentileProbability=accuracy(percentileModel,testPerFeatures)
originalValue=list(cleanTestData[0])
roc(originalValue,percentileProbability)


# In[ ]:


#This is for TF-IDF using Min-Df
mindfModel=generateModel(trainFeatures)
accuracyForMinDf,probabilityForMinDf=accuracy(mindfModel,testFeatures)
originalValue=list(cleanTestData[0])
roc(originalValue,probabilityForMinDf)


# In[ ]:


#This is for select percentile using Min-Df
mindfPercentileModel=generateModel(trainPerFeatures)
accuForMinDf,probaForMinDf=accuracy(mindfPercentileModel,testPerFeatures)
originalValue=list(cleanTestData[0])
roc(originalValue,probaForMinDf)


# In[167]:


#This is for Bigram
(original,tested,probability)=testData(cleanTestData.copy(),"bi")
roc(original,probability)


# # __Part E: Feature Selection__

# In[ ]:


#One feature bigram is already done earlier
#Here nltk pos_tag will be used


# In[ ]:


#It would be done on the cleaned Data
import nltk 
nltk.download('averaged_perceptron_tagger')

taggedList=[]
dataForTagging=cleanTrainData[5].to_numpy()

for sentences in tqdm(dataForTagging):
    wordList =  sentences.split()
    taggedFinal = nltk.pos_tag(wordList)
    taggedList.append(taggedFinal)
#     break
    
# with open('taggedSentences','wb') as f:
#     pickle.dump(taggedList,f)


# In[18]:


# with open('taggedSentences','rb') as f:
#     finalTaggedList = pickle.load(f)

print(finalTaggedList[0])


# In[ ]:


trainNegative=cleanTrainData[cleanTrainData[0]==0][5]
trainPositive=cleanTrainData[cleanTrainData[0]==4][5]

positiveTaggedList=[]
for sentences in tqdm(trainPositive):
#     print(sentences)
    wordList =  sentences.split()
    taggedFinal = nltk.pos_tag(wordList)
    positiveTaggedList.append(taggedFinal)
    
negativeTaggedList=[]
for sentences in tqdm(trainNegative):
#     print(sentences)
    wordList =  sentences.split()
    taggedFinal = nltk.pos_tag(wordList)
    negativeTaggedList.append(taggedFinal)
    
# with open('positiveTagged','wb') as f:
#     pickle.dump(positiveTaggedList,f)
    
# with open('negativeTagged','wb') as f:
#     pickle.dump(negativeTaggedList,f)


# In[5]:


# with open('positiveTagged','rb') as f:
#     positiveTaggedList = pickle.load(f)


# In[6]:


# with open('negativeTagged','rb') as f:
#     negativeTaggedList = pickle.load(f)
# # print(len(positiveTaggedList))

