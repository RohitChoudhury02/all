#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('http://192.168.12.90/Exam/MLLAB_AIML/Day3/hour.csv')
df.head()


# ## EDA & Visualization

# In[3]:


plt.figure(figsize=(12,6))
sns.countplot(x='hr', data=df) #, hue='hr', palette='viridis',legend=False
plt.title('Hourly Bike Sharing Counts')
plt.xlabel('Hour of the Day')
plt.ylabel('Count')
plt.show()


# In[4]:


plt.figure(figsize=(12,6))
sns.histplot(df['cnt'], kde=True, label='Total Count', color='blue')
sns.histplot(df['casual'], kde=True, label='Casual Count', color='orange')
sns.histplot(df['registered'], kde=True, label='Registered Count', color='green')
plt.title('Distribution of Count, Casual, and Registered')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[5]:


plt.figure(figsize=(12,6))
sns.countplot(x='weekday', hue='holiday', data=df, palette='Set2')
plt.title('Weekday-wise Bike Sharing Counts with Holiday Indicator')
plt.xlabel('Weekday')
plt.ylabel('Count')
plt.legend(title='Holiday', loc='upper right', labels=['Not Holiday', 'Holiday'])
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(x='weekday', hue='workingday', data=df, palette='Set2')
plt.title('Weekday-wise Bike Sharing Counts with Wprking Day Indicator')
plt.xlabel('Weekday')
plt.ylabel('Count')
plt.legend(title='Working Day', loc='upper right', labels=['Non-Working Day', 'Working Day'])
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(x='mnth', y='cnt', hue='yr', data=df, palette='coolwarm', errcolor=None)
plt.title('Month-wise Bike Sharing Counts for 2011 & 2012')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()


# In[6]:


#numeric_columns = df.select_dtypes(include=np.number)
correlation_matrix = df.corr()

plt.figure(figsize=(15,15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidth=.5)
plt.title('Correlation Matrix')
plt.show()


# ## Pre-Processing & Data Engineering

# In[8]:


import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

cont = ['temp', 'atemp', 'hum', 'windspeed']
cat = ['season', 'yr', 'hr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
scaler = MinMaxScaler()
encoder = OneHotEncoder(sparse=False)

x_scaled = scaler.fit_transform(df[cont])
x_encoded = encoder.fit_transform(df[cat])

x_combined = pd.concat([pd.DataFrame(x_scaled), pd.DataFrame(x_encoded)],axis=1)

y_train=df['cnt']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_combined, y_train, test_size = 0.2, random_state = 42)


# ## Linear Regression by finding the Co-efficients using below approaches.

# In[9]:


import numpy as np
from scipy.linalg import lstsq

x_linear = df[['temp', 'atemp', 'hum', 'windspeed']].values
y_linear = df['cnt'].values

x_linear = np.column_stack((np.ones(len(x_linear)),x_linear))

theta, residuals, rank, s = lstsq(x_linear, y_linear)

print("Co-efficient (Theta): ", theta)


# ## Linear Regression using sklearn

# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

linear_model = LinearRegression()

linear_model.fit(x_train, y_train)

y_pred = linear_model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

r2 = r2_score(y_test, y_pred)
print("R2 Score: ", r2)

