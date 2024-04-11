#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[4]:


oneDarr = np.array([1,2,3,4,5])


# In[5]:


print(oneDarr)


# In[7]:


twoDarr = np.array([[1,2,3], [4,5,6], [7,8,9]])


# In[8]:


print(twoDarr)


# In[11]:


arr10_5 = np.full(10, 5)


# In[12]:


print(arr10_5)


# In[13]:


arr30_0 = np.full(30, 0)


# In[14]:


print(arr30_0)


# In[16]:


oddArr = np.arange(21,41,2)


# In[17]:


print(oddArr)


# In[18]:


identityMat = np.eye(5,5)


# In[19]:


print(identityMat)


# In[21]:


randomNum = np.random.randn(20)


# In[22]:


print(randomNum)


# In[24]:


list=[1,2,3,4,5,6,7,8,9]
array = np.array(list)
print(array)


# In[25]:


random2DArr = np.random.randn(4,4,4)


# In[26]:


print(random2DArr)


# In[27]:


random2DArr2 = np.random.randn(10,10,10)


# In[28]:


print(random2DArr2)


# In[30]:


max = np.max(random2DArr2)
min = np.min(random2DArr2)
print("Maximum: ", max)
print("Minimum: ", min)


# In[31]:


checkboard = np.tile(8,8)


# In[36]:


pat = np.array([[1,0], [0,1]])
checkboard = np.tile(pat, (4,4))
print(checkboard)


# In[39]:


A = np.array([1,2,3])
B = np.array([1,1,1])

print("Array A: ", A)
print("Array B: ", B)


# In[47]:


print("A+B: ", A+B)
print("A-B: ", A-B)
print("A*B: ", A*B)
print("A/B: ", A/B)
print("A**3: ", A**3)

A=np.arange(1,6)
B=np.arange(6,15,2)

print("A: ", A)
print("B: ", B)


# In[57]:


sampleArr = np.array([1,45,60,90])
print(sampleArr)

sin = np.sin(np.radians(sampleArr))
print("Sin of the given array: ", sin)

cosine = np.cos(np.radians(sampleArr))
print("Cosine of the given array: ", cosine)

tan = np.tan(np.radians(sampleArr))
print("Tan of the given array: ", tan)


# In[64]:


X = np.array([-180., -90., 90., 180.])
print(X)

print("Radian = ", np.radians(X))


# In[66]:


Y = np.array([4.1, 2.5, 44.5, 25.9, -1.1, -9.5, -6.9])
print(Y)


# In[68]:


print("Round of Y: ", np.round(Y))
print("Ceil of Y: ", np.ceil(Y))
print("Floor of Y: ", np.floor(Y))


# In[72]:


Mat_X = np.array([1,2,3])
Mat_Y = np.array([4,5,6])

print("Mat X: ", Mat_X)
print("Mat Y: ", Mat_Y)

print("Divide of X & Y: ", np.divide(Mat_X, Mat_Y))
print("True Divide of X & Y: ", np.true_divide(Mat_X, Mat_Y))
print("Floor Divide of X & Y: ", np.floor_divide(Mat_X, Mat_Y))


# In[73]:


Mat_A = np.array([1,2,3])
Mat_B = np.array([4,5,6])
print("Multiplication of A & B Matrix: ", Mat_A*Mat_B)


# In[77]:


X = np.array([8,4,22,3,66,12,1,5])
print(X)
X[X<5] = 4
X[X>9] = 8
print("After replacements: ", X)


# In[84]:


X = np.array([8,4,22,3,66,12,1,5])
print("Length of X: ", len(X))
print("Shape of X: ", np.shape(X))
print("Dimension of X: ", X.ndim)


# In[86]:


twoD = np.arange(-1, 14, 0.25).reshape(-1,1)
print(twoD)


# In[95]:


X = np.arange(-1, 14, 0.25).reshape(-1,1)

print("Sum of all elements of X: ", np.sum(X))

print("Sum of all elements of X row-wise: ", np.sum(X, axis=1))
print("Sum of all elements of X column-wise: ", np.sum(X, axis=0))

print("Max of all elements of X: ", np.max(X))
print("Min of all elements of X: ", np.min(X))

print("Max of all elements of X row-wise: ", np.max(X, axis=1))
print("Min of all elements of X col-wise: ", np.min(X, axis=0))

print("Standard Deviation of all elements of X col-wise: ", np.std(X, axis=0))

