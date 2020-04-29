
# coding: utf-8

# In[1]:


import csv
import time
import numpy as np
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



xList = np.loadtxt('linearX.csv', delimiter='\n')
yList = np.loadtxt('linearY.csv', delimiter='\n')
thetaZPlot=[]
thetaOPlot=[]
costPlot=[]

print("xlist before")
# print(xList)
mean = np.mean(xList)
sD= np.std(xList)
for i in range(len(xList)):
    xList[i] = (xList[i]-mean)/sD
print("xlist after")
# print(xList)

# plt.scatter(xList,yList)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
temp=1
eta = 0.001
x=np.array([np.ones(100),xList])
print("x is")
# print(x)
print(x.shape)
y=yList.reshape(100,1)
theta = np.zeros(2)
# thetaZPlot.append(0)
# thetaOPlot.append(0)
theta = theta.reshape(2,1)


def costFun(m,theta,x):
    prod=np.dot(theta.T,x)
    diff=y-prod.T
    out=np.square(diff)
    return (1/(2*m))*np.sum(out)


def gradient(y,theta,x):
    global temp
    prod=np.dot(theta.T,x)
    diff=y-prod.T
    ref1=np.copy(theta[0][0])
    ref2=np.copy(theta[1][0])
    theta[0][0]=ref1+(eta*np.sum(diff))/100
    xlistRef=  xList.reshape(100,1)
    check = np.dot(diff.T,xlistRef)
    theta[1][0]=ref2+(eta*check[0][0])/100
    temp+=1

IC=costFun(100,theta,x)
# costPlot.append(IC)
gradient(y,theta,x)
thetaZPlot.append(theta[0][0])
thetaOPlot.append(theta[1][0])
FC=costFun(100,theta,x)
costPlot.append(FC)

start = time.time()

print(abs(IC-FC))

while(abs(IC - FC) > 1e-15):
    IC=FC
    gradient(y,theta,x)
    thetaZPlot.append(theta[0][0])
    thetaOPlot.append(theta[1][0])
    FC=costFun(100,theta,x)
    costPlot.append(FC)

print("final theta is")
print(theta)
newY=np.dot(theta.T,x).T


# In[2]:


print(xList)
plt.plot(xList.reshape(100,1),y,"bo")
plt.plot(xList.reshape(100,1),newY,color='k')
plt.ylabel('Density of wine')
plt.xlabel('Acidity of Wine')
# figasd = plt.figure()
# axasd = figasd.add_subplot(111)
# print(type(axasd))
plt.show()


# In[3]:


def mesh_support(thetaZSupp,thetaOSupp,m):
   output=[]
   (zTotal,) = thetaZSupp.shape
   (oTotal,) = thetaOSupp.shape
#     print(x.shape)
#     print(zTotal)
   a=0
   b=0

   while(a<zTotal and b<oTotal):
       thetaSupp=np.array([thetaZSupp[a],thetaOSupp[b]])
       thetaSupp=thetaSupp.reshape(1,2)
       prodSupp=np.dot(thetaSupp,x)
       diffSupp=prodSupp.T-y
       outSupp=np.square(diffSupp)
       output.append((1/(2*m))*np.sum(outSupp))
       a+=1
       b+=1
   return output


# In[4]:


# ref = np.array([0.33,-0.66])
# a=np.array([0.804048])
# b=np.array([-1.26369])
# ref=ref.reshape(2,1)
# print(ref)
# x1=costFun(100,ref,x)
# x2=mesh_support(a,b,100)
# print(x1)
# print(x2)


# In[5]:



def contourplot():
#     %matplotlib qt
    fig1 = plt.figure()
    axis1 = fig1.add_subplot(111)
    axis1.set_xlabel('Theta0')
    axis1.set_ylabel('Theta1')
    line1, = axis1.plot([],[], color='black')#,lw=1,color='black',markersize=5)
    def ani(i):
        xAxis1=thetaZPlot
        yAxis1=thetaOPlot
        line1.set_data(xAxis1[:i],yAxis1[:i])
        return line1,

    t0=np.linspace(-1e-6,2,100)
    t1=np.linspace(-1,1,100)
    X1,Y1=np.meshgrid(t0,t1)
    zSupp=np.asarray(mesh_support(X1.flatten(),Y1.flatten(),100))
    Z1=zSupp.reshape(X1.shape)
    a=axis1.contourf(X1,Y1,Z1,cmap=cm.viridis, alpha=1)

    length1=len(thetaZPlot)
    plottingAnimation = animation.FuncAnimation(fig1, ani,interval=200,repeat=True)#,blit='False',repeat='False')
    plt.show()
    return a, plottingAnimation

(a,b) = contourplot()


# In[6]:


def meshplot():
#     %matplotlib qt
    fig2 = plt.figure()
    axis2 = fig2.add_subplot(111, projection='3d')
    axis2.set_xlabel('Theta0')
    axis2.set_ylabel('Theta1')
    axis2.set_zlabel('Cost')
    def anime(i):
        xAxis2=thetaZPlot
        yAxis2=thetaOPlot
        line2.set_data(xAxis2[:i],yAxis2[:i])
        line2.set_3d_properties(costPlot[:i])
        return line2,

    t3=np.linspace(-1e-6,2,100)
    t4=np.linspace(-1,1,100)
    X2,Y2=np.meshgrid(t3,t4)
    zSupp2=np.asarray(mesh_support(X2.flatten(),Y2.flatten(),100))
    print(zSupp2)
    Z2=zSupp2.reshape(X2.shape)
    c=axis2.plot_surface(X2,Y2,Z2,cmap=cm.viridis, alpha=1)
    line2, = axis2.plot([],[],[],lw=1,color='black',markersize=5)
    length2=len(thetaZPlot)
    plotAnimation = animation.FuncAnimation(fig2, anime,interval=200,repeat='True')
    plt.show()
    return c,plotAnimation

(k,l)=meshplot()
