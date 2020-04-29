
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import time
import csv
from mpl_toolkits.mplot3d import Axes3D
x0=np.ones(1000000)
x1=np.random.normal(3,2,1000000)
print(x1.shape)
x2=np.random.normal(-1,2,1000000)
x=np.array((x0,x1,x2),dtype=float)
# print(x.shape)
# print(x)
thetasObtained=[]


# In[2]:


theta=np.array([3,1,2])
theta=theta.reshape(3,1)
thetaN=np.zeros((3,1))
print(theta)
print(thetaN.shape)
# print(thetaN[2][0])


# In[3]:


# print("-------------------------------")
hyp=np.dot(theta.T,x)
# print(hyp)
# print(hyp.shape)
# print("--------------------------------")


hyp1=np.random.normal(loc=hyp,scale=math.sqrt(2))
# print(hyp1.shape)
# print(hyp1)
# print(hyp1[0])
# print(hyp1[0].shape)
y=hyp1.T


dataToShuffle=np.append(x,y.T,axis=0)
# print("initially")
# print(dataToShuffle)
dataToShuffle=dataToShuffle.T
# print("finally")
# print(dataToShuffle)
# print(dataToShuffle.shape)

print("shuffling data now")
np.random.shuffle(dataToShuffle)
print(dataToShuffle)



print("now getting data to the required form")
x=dataToShuffle[:,0:3].T
y=dataToShuffle[:,3:4]
# print(x)
# print(x.shape)
# print(y)
# print(y.shape)


# plt.scatter(x1,hyp1[0])
# plt.scatter(x2,hyp1[0])
# plt.show()


# In[4]:


#Sampling done above
#This is the space for debugging
# print(x)
# print(x.shape)
(x[:,:1])
i=0
print(theta.shape)
# print(prod.T.shape)
print(theta.T.shape)
# print(x[:,i:i+1000])
print(x[:,i:i+1000].shape)
prod=np.dot(theta.T,x[:,i:i+1000])
print(prod.shape)
diff=y[i:i+1000]-prod.T
print(diff.shape)
x[1]

print("printing x here" , x)
print("printing x ka shape", x.shape)
print("printing y  here", y)
print("printing y ka shape", y.shape)


# In[5]:


def costFun(mPara,yPara,thetaPara,xPara):
    prod=np.dot(thetaPara.T,xPara)

    diff=yPara-prod.T

    out=np.square(diff)
    return (1/(2*mPara))*np.sum(out)


# In[6]:


theta0ForPlot,theta1ForPlot,theta2ForPlot=[0],[0],[0]


# In[7]:


def stochGrad(yRef,thetaN,xRef,bSize,eta):

    prod=np.dot(thetaN.T,xRef)
    diff=yRef-prod.T

    ref1=np.copy(thetaN[0][0])
    ref2=np.copy(thetaN[1][0])
    ref3=np.copy(thetaN[2][0])

    thetaN[0][0]=ref1+(eta*np.sum(diff))/bSize

    (a,b)=xRef.shape


    check1 = np.dot(diff.T,xRef[1].reshape(b,1))
    check2 = np.dot(diff.T,xRef[2].reshape(b,1))
#     print(check1.shape)

    thetaN[1][0]=ref2+(eta*check1[0][0])/bSize
    thetaN[2][0]=ref3+(eta*check2[0][0])/bSize
    theta0ForPlot.append(thetaN[0][0])
    theta1ForPlot.append(thetaN[1][0])
    theta2ForPlot.append(thetaN[2][0])


# In[8]:


def originStoch(bSize,convergeCriteria):
    eta=0.001
    minmIter=1000
    costInit,costFinal,i,costcheckI,costcheckF=0,0,0,0,0
    start=time.time()

    count=0

    for j in range(2*minmIter):
        costDummy=costFun(bSize,y[i:i+bSize],thetaN,x[:,i:i+bSize])
        temp=stochGrad(y[i:i+bSize],thetaN,x[:,i:i+bSize],bSize,eta)
        i=(i+bSize)%1000000
        if(j<minmIter):
            costInit+=costDummy
        else:
            costFinal+=costDummy
    count+=(2*minmIter)

    costcheckI,costcheckF=costInit/minmIter,costFinal/minmIter
#     print(costcheckI)
#     print(costcheckF)

    while(True):
        costSupp=0

        if(abs(costcheckI-costcheckF) < convergeCriteria):
            break
        for k in range(minmIter):
            costDummy=costFun(bSize,y[i:i+bSize],thetaN,x[:,i:i+bSize])
            stochGrad(y[i:i+bSize],thetaN,x[:,i:i+bSize],bSize,eta)
            i=(i+bSize)%1000000
            costSupp+=costDummy
            count+=1
        costInit=costFinal
        costFinal=costSupp
        costcheckI = costInit/minmIter
        costcheckF = costFinal/minmIter

    finalTime=time.time()-start
    print("Time it took to converge is ")
    print(finalTime)
    print("Number of iterations is")
    print(count)
    return thetaN

#originStoch take batch size and convergeCriteria as parameters
thetaN=np.zeros((3,1))
thetaFromHyp=originStoch(10000,1e-5)
print(thetaFromHyp)
thetasObtained.append(thetaFromHyp)


# In[9]:


# print(theta0ForPlot)
# print(theta1ForPlot)
# print(theta2ForPlot)
def plotting():
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    line, = axis.plot([],[],[],lw=1,color='black',markersize=5)
    axis.set_xlabel('Theta0')
    axis.set_ylabel('Theta1')
    axis.set_zlabel('Theta2')
    axis.set_xlim(-0.5, 3.5)
    axis.set_ylim(-0.5,2)
    axis.set_zlim(-0.5,2.5)
    def animate(i):
        line.set_data(theta0ForPlot[:i],theta1ForPlot[:i])
        line.set_3d_properties(theta2ForPlot[:i])
        return line,
    plotAnimation = animation.FuncAnimation(fig, animate,interval=0.00001,repeat=False)
    plt.show()
    return plotAnimation
plotting()
# # theta0ForPlot,theta1ForPlot,theta2ForPlot=[0],[0],[0]


# In[10]:


# theta=np.array([3,1,2])
# theta=theta.reshape(3,1)
# print(costFun(1000000,y,theta,x))


# In[11]:


load=pd.read_csv("q2test.csv")
testData=np.array(load)
yTest=testData[:,2:3]
xTestInit=testData[:,0:2].T
xTest=np.array((np.ones(10000),xTestInit[0],xTestInit[1]))
print(xTest,xTest.shape)
print(yTest,yTest.shape)

def testStoch(thetaTestPar,xTestPar,yTestPar):
    prod=np.dot(thetaTestPar.T,xTestPar).T
    err=(yTestPar-prod)**2
    error=np.sum(err)
    return error/(2*yTestPar.size)

thetaForTesting=np.array([[3],[1],[2]])
print(thetaForTesting)
error = testStoch(thetaForTesting,xTest,yTest)
print(error)
