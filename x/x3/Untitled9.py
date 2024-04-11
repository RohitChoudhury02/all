#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivatives(x):
    return x*(1-x)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

input_neurons=2
hidden_neurons=2
output_neurons=1
hidden_weights=np.random.uniform(size=(input_neurons,hidden_neurons))
hidden_bias=np.random.uniform(size=(1,hidden_neurons))
output_weights=np.random.uniform(size=(hidden_neurons,output_neurons))
output_bias = np.random.uniform(size=(1,output_neurons))
lr=0.3
epochs=10000


for epoch in range(epochs):
    hidden_layer_activation=np.dot(X,hidden_weights)+hidden_bias
    print(f"epoch {epoch+1},Hidden Weights : ",hidden_weights)
    hidden_layer_output=sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output,output_weights)+output_bias
    
    predicted_output=sigmoid(output_layer_activation)
    error=y-predicted_output
    print(f"epoch {epoch+1},Output Weights : ",hidden_weights)
    print(f"epoch {epoch+1},Prediction : ",predicted_output)
    
    d_predicted_output= error*sigmoid_derivatives(predicted_output)
    error_hidden_layer=d_predicted_output.dot(output_weights.T)
    d_hidden_layer=error_hidden_layer*sigmoid_derivatives(hidden_layer_output)
    
    output_weights+=hidden_layer_output.T.dot(d_predicted_output)*lr
    output_bias+=np.sum(d_predicted_output,axis=0,keepdims=True)*lr
    hidden_weights+=X.T.dot(d_hidden_layer)*lr
    hidden_bias+=np.sum(d_hidden_layer,axis=0,keepdims=True)*lr

    
    


# In[8]:


final_predictions=predicted_output
print(np.round(final_predictions,2))


# In[10]:


import pandas as pd
text_data = pd.read_csv("E:\Mainak Das_087\Day6\FullIJCNN2013.csv",header=None)


# In[11]:


text_data.head()


# In[12]:


text_data


# In[26]:


from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('E:\Mainak Das_087\Day6\FullIJCNN2013/00001.ppm')
image.resize(size=(500,500))


# In[23]:


image_array=np.array(image)
image_array


# In[24]:


text_data[5].value_counts()


# In[25]:


distribution = text_data[5].value_counts()
plt.bar(distribution.index,distribution.values)
plt.xlabel('Class')
plt.ylabel('Freq')
plt.show()

