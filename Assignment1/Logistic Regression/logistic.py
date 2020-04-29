
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time


# In[2]:


loadX=pd.read_csv("logisticX.csv",header=None)
loadY=pd.read_csv("logisticY.csv",header=None)
loadX=np.array(loadX)
# print(loadX)
X1 = ((loadX - loadX.mean(axis=0))/loadX.std(axis=0)).T
# print(X1)
# print(X1[0])
x=np.array((np.ones(100),X1[0],X1[1])).T
# print(x)
# X = np.vstack((np.ones(X1.shape[0]), X1))


# In[3]:


x.shape


# In[4]:


# print("printing after normalization")

# print(x)
# print(x.shape)

y=np.array(loadY)

# print(y.shape)
theta=np.zeros((3,1))
# print(theta)


# In[5]:


def gZ(x):
    global theta
    temp=np.dot(x,theta)
    out = [1/(1+math.exp(-1*i)) for i in temp]
    out=np.asarray(out)
    out=out.reshape(100,1)
    return out


# In[6]:


def hessian(x,gz):
    temp1=gz
    temp2=1-gz

    diagonal=temp1*temp2
    diagonal=diagonal.T


    req=np.diag(diagonal[0])
    temp3=np.dot(x.T,req)
    temp4=np.dot(temp3,x)

    hessian=-1*temp4

    return hessian


# In[7]:


def gradient(y,gz,x):
    temp1=y-gz
    temp=temp1.T
    prod=np.dot(temp,x)
    return prod.T


# In[8]:


def logisticRegression(x,y):
    global theta
    theta=np.zeros((3,1))
    for i in range(8):
        gzCall=gZ(x)
#         print(gzCall.shape)
        hess=hessian(x,gzCall)
#         print(hess.shape)
        hessInv=np.linalg.inv(hess)
#         print(hessInv.shape)
        grad=gradient(y,gzCall,x)
#         print(grad.shape)
        theta1=theta-np.dot(hessInv,grad)
        theta=theta1
    print(theta)
    print(theta.shape)

logisticRegression(x,y)


# In[9]:


def plotting():
#     print(x)
    print(theta)

    temp1=np.array((X1[0],X1[1],y.T[0])).T
    print(temp1.shape)
    temp2=temp1.T
    print(temp2.shape)

    positiveX1=temp1[temp1[:,2] == 1]
    negativeX1=temp1[temp1[:,2] == 0]
    positiveX=positiveX1[:,:2].T

    negativeX=negativeX1[:,:2].T

    fig, ax = plt.subplots()
    ax.plot(positiveX[0],positiveX[1],"bo",color='red',marker='o',label='Class 1')
    ax.plot(negativeX[0],negativeX[1],"bo",color='blue',marker='*',label='Class 0')
    ax.set_ylabel('X2')
    ax.set_xlabel('X1')
    plotLineX=np.linspace(-3,3,100)
    plotLineX1=np.array((np.ones(100),plotLineX)).T
#     print(plotLineX)
    thetaForPlot=theta[:2,:]
#     print(thetaForPlot)
    prod1 = -1*np.dot(plotLineX1,thetaForPlot)
#     print("prod1 ka shape")
#     print(prod1.shape)
#     print("printing theta 2 0")
#     print(theta[])
    prod = prod1.T /theta[2][0]
#     print("prod ka shape")
#     print(prod.shape)
    plotLineY=prod[0]
#     print("plotLineX.shape")
#     print(plotLineX.shape)

#     print("finally printing the shape  of X")
#     print(plotLineX.shape)
#     print("finally printing the shape of Y")
#     print(plotLineY.shape)

    ax.plot(plotLineX,plotLineY,color='black',label='Decision Boundary')
    ax.legend(loc='upper left')
    plt.show()

#     hypothesis1 = np.asarray(hypothesis).reshape(100,1)
#     print(x)
#     print(x.shape)
#     print(hypothesis1)
#     plt.scatter(X1,y)
#     plt.scatter(X1,hypothesis1)
#     plt.show()

#     print(hypothesis)
plotting()
