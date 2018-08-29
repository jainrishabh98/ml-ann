
# coding: utf-8

# In[296]:


#IMPLEMENTATION OF ARTIFICIAL NEURAL NETWORK

import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler , OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt


# In[297]:


bank = np.array(pd.read_csv('path_to_Churn_Modelling.csv'))
le = LabelEncoder()
one = OneHotEncoder(sparse=False)
bank[:,4]=le.fit_transform(bank[:,4])
bank[:,5]=le.fit_transform(bank[:,5])
g = one.fit_transform(bank[:,4].reshape((10000,1)))
h = one.fit_transform(bank[:,5].reshape((10000,1)))
scaler = StandardScaler()
inputs = np.array([bank[:,3],g[:,0],g[:,1],g[:,2],h[:,0],h[:,1],bank[:,6],bank[:,7],bank[:,8],bank[:,9],bank[:,10],bank[:,11],bank[:,12]])
inputs = inputs.T
inputs = inputs.astype(float)
inputs = scaler.fit_transform(inputs)
target = bank[:,13].astype(float)
x, xt, y, yt = train_test_split(inputs, target, test_size=0.2, random_state=0)
ovr = np.array([x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11],x[:,12],y])
ovr = ovr.T
print(inputs.shape)
print(x.shape)
y=y.reshape((8000,1))
print(y.shape)
#print(g.shape)
#print(h.shape)
#print(ovr.shape)


# In[334]:


w1=np.random.randn(13,8) 
w2=np.random.randn(8,3)
w3=np.random.randn(3,1)
b1=np.zeros((1,8))
b2=np.zeros((1,3))
b3=np.zeros((1,1))
Ew1 = np.zeros((13,8))
Ew2 = np.zeros((8,3))
Ew3 = np.zeros((3,1))
Eb1 = np.zeros((1,8))
Eb2 = np.zeros((1,3))
Eb3 = np.zeros((1,1))
alpha=50
e=1e-8
m=8000
batch_size=512 
J=0
L=0
v=alpha/m
ch=[]
it=[]
k=0


# In[335]:


for itr in range(0,170): #Applying Mini-Batch Gradient Descent with RMSProp Optimisation
    np.random.shuffle(ovr)
    for i in range(0,int(m/batch_size)):
        xb = ovr[(i*batch_size):((i+1)*batch_size),0:13]
        yb = ovr[(i*batch_size):((i+1)*batch_size),13:]
        k=k+1
        #Forward Propagation
        z1 = b1 + xb.dot(w1)
        a1 = 1/(1+np.exp(-z1))  
        z2 = b2 + a1.dot(w2)
        a2 = 1/(1+np.exp(-z2))  
        z3 = b3 + a2.dot(w3)
        a3 = 1/(1+np.exp(-z3))  
    
        #Calculating Loss And Overall Cost
        L= -(yb*np.log(a3 + e) + (1-yb)*np.log(1-a3 + e)) 
        J = np.sum(L)/m
    
        #Backpropagation
        dz3 = (a3 - yb)
        dw3 = (a2.T).dot(dz3)
        Ew3 = 0.9*Ew3 + 0.1*(dw3**2)
        db3 = dz3.sum()
        Eb3 = 0.9*Eb3 + 0.1*(db3**2)
        dz2 = dz3.dot(w3.T) * (a2 * (1-a2)) 
        dw2 = (a1.T).dot(dz2)
        Ew2 = 0.9*Ew2 + 0.1*(dw2**2)
        db2 = dz2.sum(axis=0,keepdims=True)
        Eb2 = 0.9*Eb2 + 0.1*(db2**2)
        #print(db2.shape)
        dz1 = dz2.dot(w2.T) * (a1 * (1-a1))
        dw1 = (xb.T).dot(dz1) #x can be seen as a0
        Ew1 = 0.9*Ew1 + 0.1*(dw1**2)
        db1 = dz1.sum(axis=0,keepdims=True)
        Eb1 = 0.9*Eb1 + 0.1*(db1**2)
    
        #Updating the Weights
        w3 = w3 - (dw3/(np.sqrt(Ew3)+e))*v
        b3 = b3 - (db3/(np.sqrt(Eb3)+e))*v
        w2 = w2 - (dw2/(np.sqrt(Ew2)+e))*v
        b2 = b2 - (db2/(np.sqrt(Eb2)+e))*v
        w1 = w1 - (dw1/(np.sqrt(Ew1)+e))*v
        b1 = b1 - (db1/(np.sqrt(Eb1)+e))*v
    
        ch.append(J)
        it.append(k)

z1 = b1 + x.dot(w1)
a1 = 1/(1+np.exp(-z1))  
z2 = b2 + a1.dot(w2)
a2 = 1/(1+np.exp(-z2))  
z3 = b3 + a2.dot(w3)
a3 = 1/(1+np.exp(-z3)) 
mx = a3.max()
mn = a3.min()
#print(mx)
#print(mn)
itt = []
for i in range(0,8000):
    itt.append(i+1)
L= -(y*np.log(a3 + e) + (1-y)*np.log(1-a3 + e)) 
J = np.sum(L)/m
plt.figure()
plt.xlabel("Training Example")
plt.ylabel("Predictions")
plt.scatter( itt, a3)
plt.figure()
plt.xlabel("Iteration")
plt.ylabel("Total Loss")
plt.plot(it,ch)
plt.show()
c1=0
c2=0
sum=0
for i in range(0,8000):
    if a3[i]>=(0.65*(mx-mn))+mn:  
        a3[i]=1
        c1=c1+1
    else:
        a3[i]=0
        c2=c2+1
    sum+=abs(y[i]-a3[i])
    
acc = 1 - (sum / 8000)
print(acc) 
print(c1)
print(c2)


# In[336]:


z1 = b1 + xt.dot(w1)
a1 = 1/(1+np.exp(-z1))
z2 = b2 + a1.dot(w2)
a2 = 1/(1+np.exp(-z2))  
z3 = b3 + a2.dot(w3)
a3 = 1/(1+np.exp(-z3))
c1=0
c2=0
sum=0
for i in range(0,2000):
    if a3[i]>=(0.65*(mx-mn))+mn:  
        a3[i]=1
        c1=c1+1
    else:
        a3[i]=0
        c2=c2+1
    sum+=abs(yt[i]-a3[i])
    
acc = 1 - (sum / 2000)
print(c1)
print(c2)
print(acc)

