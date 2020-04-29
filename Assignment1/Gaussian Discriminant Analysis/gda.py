
# coding: utf-8

# In[1]:


import numpy as np
import math
from math import log
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


xLoad = np.loadtxt( 'q4x.dat' )
# print(xLoad)
# print(x)
yStr = np.loadtxt('q4y.dat',dtype='str')
yNmr = np.asarray([0  if x=='Alaska' else 1 for x in yStr])

# print(yNmr)

xs = ((xLoad - xLoad.mean(axis=0))/xLoad.std(axis=0))
# print(norX)
xWithIcpt=np.array((np.ones(100),xs.T[0],xs.T[1])).T

classifyMatrix = np.array((xWithIcpt.T[1],xWithIcpt.T[2],yNmr)).T
# print(classifyMatrix)

positiveClass1=classifyMatrix[classifyMatrix[:,2] == 1]
negativeClass1=classifyMatrix[classifyMatrix[:,2] == 0]

# print("printing positive class")
positiveClass=positiveClass1[:,:2]
negativeClass=negativeClass1[:,:2]

print("printing postive and negative class")
# print(positiveClass)
# print(negativeClass)
# negativeX1=temp1[temp1[:,2] == 0]
# classifyMatrix0 =

# print(x)
# print(xWithIcpt.shape)
#Assigning 0 to Alaska and 1 to Canada. will use it if needed in the future


# In[ ]:





# In[3]:


# print("initial x")
# print(xs[:5])

def phi(Y):
    number=np.count_nonzero(Y==1)
    a=Y.size
    return number/a

phii=phi(yNmr)
# print(phii)


def mu0(X,Y):

    number=np.count_nonzero(Y==0)

    paraY=np.asarray([0 if p==1 else 1 for p in Y])

    prodY=(np.dot(paraY,X)).reshape(2,1)

    out=prodY/number
    return out

# print("printing x and yNmr")
# print(x[:5])
# print(yNmr[:5])


def mu1(X,Y):
    number=np.count_nonzero(Y==1)
    prodY=np.dot(Y.T,X).reshape(2,1)
    out=prodY/number
    return out

# print("printing bahar se")
# print(mu0(x,yNmr))
# print(mu0(x,yNmr))
# print(mu0(x,yNmr))
# print("printing small x")
# print(xs[:5])

muu0=mu0(xs,yNmr)


# print("printing mu0 and mu1")
print(muu0)
muu1=mu1(xs,yNmr)
print(muu1)

def covariance(X,Y,muuuu0,muuuu1):
    xrefCov=np.copy(X)
    print("printing xref Copy inside")
#     print(xrefCov)
    yrefCov=np.copy(Y)
#     print("printing x inside covariance at start")
#     print(xs[:5])
    length=Y.size
#     print("printing length", length)
    for i in range(length):
        if(yrefCov[i]==1):
            xrefCov[i]=xrefCov[i]-muuuu1.T
        else:
            xrefCov[i]=xrefCov[i]-muuuu0.T
    prod=(np.dot(xrefCov.T,xrefCov))/length
#     print("printing x inside covariance at last")
#     print(xs[:5])
    return prod

covar=covariance(xs,yNmr,muu0,muu1)
print(covar)

def decisionBoundary(phii,coVar,mU0,mU1):
    boundary=[]
    plotlineX=np.linspace(-3,3,100)
    logTerm=math.log((phii/(1-phii)),2)
    mu0Term=np.dot(np.dot(mU0.T,np.linalg.inv(coVar)),mU0)
    mu1Term=np.dot(np.dot(mU1.T,np.linalg.inv(coVar)),mU1)
    muTerm=(-1*(-1*mu0Term+mu1Term)/2)
    sigmaMu0=np.dot(np.linalg.inv(coVar),mU0)
    sigmaMu1=np.dot(np.linalg.inv(coVar),mU1)
    rhsTerm=logTerm+muTerm
    for i in range(plotlineX.size):
        temp1=plotlineX[i]*-1*(sigmaMu0[0][0]-sigmaMu1[0][0])
        temp2=rhsTerm-temp1
        temp3 = -1*(sigmaMu0[1][0]-sigmaMu1[1][0])
#         print("temp 2 is :",sigmaMu0[0][0]-sigmaMu1[0][0])
#         print("temp 3 is : ",temp3)
#         print("checking dimension of temp2",temp2.shape)
#         print("checking dimension of temp3",temp3.shape)
        boundary.append((temp2/temp3)[0][0])
#         break
    return plotlineX,boundary


outX0,outX1=decisionBoundary(phii,covar,muu0,muu1)
# print("printing outX0")
# print(outX0)
# print("printing outX1")
# print("printing x1",outX0)
# print("printing boundary")
# print(outX1)
def plotting(outX0,outX1):
    fig, ax = plt.subplots()
    ax.plot(positiveClass.T[0],positiveClass.T[1],"bo",color='red',marker='o',label='Canada [1]')
    ax.plot(negativeClass.T[0],negativeClass.T[1],"bo",color='blue',marker='*',label='Alaska [0]')
    ax.plot(outX0,outX1,color='green',label='Linear Decision Boundary')
    ax.legend()
    plt.show()

plotting(outX0,outX1)


# In[4]:


def covar0quad(MU0,X,Y):
    length=np.count_nonzero(Y==0)
    xReference = np.copy(X)
    yReference = np.copy(Y)

    for i in range(Y.size):
        if(yReference[i]==0):
            xReference[i]=xReference[i]-MU0.T
        else:
            xReference[i]-=xReference[i]

    prod=np.dot(xReference.T,xReference)/length
    return prod

covar0=covar0quad(muu0,xs,yNmr)
print(covar0)


# In[5]:


def covar1quad(MU1,X,Y):
    length=np.count_nonzero(Y==1)
    xReference = np.copy(X)
    yReference = np.copy(Y)

    for i in range(Y.size):
        if(yReference[i]==1):
            xReference[i]=xReference[i]-MU1.T
        else:
            xReference[i]-=xReference[i]

    prod=np.dot(xReference.T,xReference)/length
    return prod

covar1=covar1quad(muu1,xs,yNmr)
print(covar1)
print(np.linalg.inv(covar1))


# In[6]:


def quadraticDecisionBoundary(Phi,MU0,MU1,Covar1,Covar0):
    boundary1,boundary2=[],[]
    sigma1Inv=np.linalg.inv(Covar1)
#     print(sigma1Inv)
    sigma0Inv=np.linalg.inv(Covar0)
#     print(sigma0Inv)
    a1,b1,c1=sigma1Inv[0][0],sigma1Inv[1][1],sigma1Inv[1][0]
    a0,b0,c0=sigma0Inv[0][0],sigma0Inv[1][1],sigma0Inv[1][0]
    temp1=np.dot(MU1.T,sigma1Inv)
    p1,q1=temp1[0][0],temp1[0][1]
    temp2=np.dot(MU0.T,sigma0Inv)
    p0,q0=temp2[0][0],temp2[0][1]

    temp3=np.dot(np.dot(MU1.T,sigma1Inv),MU1)
    temp4=np.dot(np.dot(MU0.T,sigma0Inv),MU0)
    temp5=((temp3-temp4)/2)[0][0]-math.log((Phi/(1-Phi)),2)
    temp6=math.log((math.sqrt(np.linalg.det(Covar0)/np.linalg.det(Covar1))),2)
    cSupport=temp5-temp6

    plotLineX=np.linspace(-2,2,100)
    for i in range(plotLineX.size):
        a=(b1-b0)/2
        b=plotLineX[i]*(c1-c0) + (q0-q1)
        cSupp=((plotLineX[i]**2) * (a1-a0))/2 + (p0-p1)*plotLineX[i]
        c=cSupport+cSupp
#         print(a,b,c)
        boundary1.append((-1*b + math.sqrt(b**2 - 4*a*c))/(2*a))
        boundary2.append((-1*b - math.sqrt(b**2 - 4*a*c))/(2*a))
    return plotLineX,boundary1,boundary2


(t1,t2,t3)=quadraticDecisionBoundary(phii,muu0,muu1,covar1,covar0)
# print(t1)
# print(t2)
# print(t3)


# In[7]:


def plottingQuad(outX0,outX1,outX0ForQuad,outX1PD,outX1ND):
    fig, ax = plt.subplots()
    ax.plot(positiveClass.T[0],positiveClass.T[1],"bo",color='red',marker='p',label='Canada [1]')
    ax.plot(negativeClass.T[0],negativeClass.T[1],"bo",color='blue',marker='*',label='Alaska [0]')
    ax.plot(outX0,outX1,color='green',label='Linear Decision Boundary')
    ax.plot(outX0ForQuad,outX1PD,color='black',label='Quadratic Decision Boundary')
    ax.legend()
    plt.show()

plottingQuad(outX0,outX1,t1,t2,t3)


# In[8]:


#For Testing purpose


# In[9]:


print(muu1)
print(muu0)
# print(type(yNmr))
# print(yNmr)
# reqY=np.where(yNmr==1)
# print(reqY.shape)
# print(reqY)
