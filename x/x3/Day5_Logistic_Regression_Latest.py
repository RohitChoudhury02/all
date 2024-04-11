#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Reading data
df = pd.read_csv('E:\Mainak Das_087\Day4\Social_Network_Ads.csv')
df.head()


# In[3]:


# Check data info
df.info()


# In[4]:


# Separate features and labels
X = df.drop(labels=["Purchased"], axis=1)
y = df["Purchased"]


# In[5]:


# Feaure scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X


# In[6]:


# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[7]:


# Training
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)    


# In[8]:


# Prediction
y_pred = lr.predict(X_test)


# In[9]:


# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
print(f"\t\tClassification Report:\n\n{classification_report(y_test, y_pred)}")


# In[10]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=lr.classes_)
sns.heatmap(cm,annot=True,cmap=plt.cm.Blues)
plt.show()


# In[11]:


# ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

# Get FPR, TPR, Threshold
fpr, tpr, thresh = roc_curve(y_test, y_pred)

# AUC Score
auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC Score: {auc}")

# Plot curve
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, color="Blue", label="ROC Curve")
plt.plot([0,1], [0,1], color="red", label="Straight Line")
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[12]:


# Batch Gradient Descent
# Define Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Gradient Descent 
def batch_gradient_descent(X, y, epochs=20, lr=0.1):
    m, n = X.shape
    theta = np.zeros(n)
    for epoch in range(epochs):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h-y))//m
        theta -= lr*gradient
        loss = -np.mean(y*np.log(h) + (1-y)*np.log(1-h))
        print(f"Loss after epoch {epoch+1}: {loss}")
    return theta

def predict(X, theta):
    z = np.dot(X, theta)
    return np.round(sigmoid(z))

theta = batch_gradient_descent(X_train, y_train)
y_pred_batch = predict(X_test, theta)
y_pred_batch = [int(i) for i in y_pred_batch]


# In[13]:


# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred_batch)}\n")
print(f"\t\tClassification Report:\n\n{classification_report(y_test, y_pred_batch)}")


# In[14]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=lr.classes_)
sns.heatmap(cm,annot=True,cmap=plt.cm.Blues)
plt.show()


# In[15]:


# ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

# Get FPR, TPR, Threshold
fpr, tpr, thresh = roc_curve(y_test, y_pred_batch)

# AUC Score
auc = roc_auc_score(y_test, y_pred_batch)
print(f"ROC AUC Score: {auc}")

# Plot curve
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, color="Blue", label="ROC Curve")
plt.plot([0,1], [0,1], color="red", label="Straight Line")
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[16]:


y_train = y_train.values


# In[17]:


# SGD
# Define Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Stochastic Gradient Descent 
def stochastic_gradient_descent(X, y, epochs=10, lr=0.01):
    m, n = X.shape
    theta = np.zeros(n)
    for epoch in range(epochs):
        loss = 0
        for i in range(m):
            Xi = X[i, :]
            yi = y[i]
            z = np.dot(Xi, theta)
            h = sigmoid(z)
            gradient = np.dot(Xi.T, (h-yi))
            theta -= lr*gradient
            loss += -np.mean(yi*np.log(h) + (1-yi)*np.log(1-h))
        print(f"Loss after epoch {epoch+1}: {loss/m}")
    return theta

def predict(X, theta):
    z = np.dot(X, theta)
    return np.round(sigmoid(z))

theta = stochastic_gradient_descent(X_train, y_train)
y_pred_sgd = predict(X_test, theta)
y_pred_sgd = [int(i) for i in y_pred_sgd]


# In[18]:


# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred_sgd)}\n")
print(f"\t\tClassification Report:\n\n{classification_report(y_test, y_pred_sgd)}")


# In[19]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=lr.classes_)
sns.heatmap(cm,annot=True,cmap=plt.cm.Blues)
plt.show()


# In[20]:


# ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

# Get FPR, TPR, Threshold
fpr, tpr, thresh = roc_curve(y_test, y_pred_sgd)

# AUC Score
auc = roc_auc_score(y_test, y_pred_sgd)
print(f"ROC AUC Score: {auc}")

# Plot curve
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, color="Blue", label="ROC Curve")
plt.plot([0,1], [0,1], color="red", label="Straight Line")
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[21]:


# Mini Batch Gradient Descent
# Define Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Gradient Descent 
def mini_batch_gradient_descent(X, y, epochs=15, lr=0.01, batch_size=32):
    m, n = X.shape
    theta = np.zeros(n)
    for epoch in range(epochs):
        loss=0
        for batch in range(1,(X.shape[0]//batch_size)+1):
            Xi = X[batch_size*(batch-1):batch_size*batch, :]
            yi = y[batch_size*(batch-1):batch_size*batch]
            z = np.dot(Xi, theta)
            h = sigmoid(z)
            gradient = np.dot(Xi.T, (h-yi))//m
            theta -= lr*gradient
            loss += -np.mean(yi*np.log(h) + (1-yi)*np.log(1-h))
        print(f"Loss after epoch {epoch+1}: {loss/10}")
    return theta

def predict(X, theta):
    z = np.dot(X, theta)
    return np.round(sigmoid(z))

theta = mini_batch_gradient_descent(X_train, y_train)
y_pred_mini_batch = predict(X_test, theta)
y_pred_mini_batch = [int(i) for i in y_pred_mini_batch]


# In[22]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=lr.classes_)
sns.heatmap(cm,annot=True,cmap=plt.cm.Blues)
plt.show()


# In[23]:


# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(f"Accuracy: {accuracy_score(y_test, y_pred_mini_batch)}\n")
print(f"Precision: {precision_score(y_test, y_pred_mini_batch)}\n")
print(f"Recall: {recall_score(y_test, y_pred_mini_batch)}\n")
print(f"F1-score: {f1_score(y_test, y_pred_mini_batch)}\n")


# In[24]:


# ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

# Get FPR, TPR, Threshold
fpr, tpr, thresh = roc_curve(y_test, y_pred_mini_batch)

# AUC Score
auc = roc_auc_score(y_test, y_pred_mini_batch)
print(f"ROC AUC Score: {auc}")

# Plot curve
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, color="Blue", label="ROC Curve")
plt.plot([0,1], [0,1], color="red", label="Straight Line")
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.title("ROC Curve")
plt.legend()
plt.show()

