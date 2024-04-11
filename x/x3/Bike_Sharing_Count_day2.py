#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('http://192.168.12.90/Exam/MLLAB_AIML/Day3/hour.csv')
df.head()


# ## EDA & Visualization

# In[2]:


plt.figure(figsize=(12,6))
sns.countplot(x='hr', data=df) #, hue='hr', palette='viridis',legend=False
plt.title('Hourly Bike Sharing Counts')
plt.xlabel('Hour of the Day')
plt.ylabel('Count')
plt.show()


# In[3]:


plt.figure(figsize=(12,6))
sns.histplot(df['cnt'], kde=True, label='Total Count', color='blue')
sns.histplot(df['casual'], kde=True, label='Casual Count', color='orange')
sns.histplot(df['registered'], kde=True, label='Registered Count', color='green')
plt.title('Distribution of Count, Casual, and Registered')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[4]:


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


# In[5]:


correlation_matrix = df.corr()

plt.figure(figsize=(15,15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidth=.5)
plt.title('Correlation Matrix')
plt.show()


# ## Pre-Processing & Data Engineering

# In[6]:


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
x_train, x_test, y_train, y_test = train_test_split(x_combined, y_train, test_size = 0.2, random_state = 87)


# In[7]:


import numpy as np
from scipy.linalg import lstsq

x_linear = df[['temp', 'atemp', 'hum', 'windspeed']].values
y_linear = df['cnt'].values

x_linear = np.column_stack((np.ones(len(x_linear)),x_linear))

theta, residuals, rank, s = lstsq(x_linear, y_linear)

print("Co-efficient (Theta): ", theta)


# In[8]:


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


# ## Polynomial Regression for Training Data

# In[9]:


'''from sklearn.preprocessing import PolynomialFeatures 
deg = 2
poly = PolynomialFeatures(degree = deg)

x_poly = poly.fit_transform(x_train)

poly.fit(x_poly, y_train)
model = LinearRegression()

model.fit(x_poly, y_train)

x_pred2 = model.predict(x_poly)

r2b = r2_score(y_train, x_pred2)
print("R2 Score for Training Data (Polynomial): ", r2b)'''


# ## Polynomial Regression for Testing Data

# In[16]:


'''poly2 = PolynomialFeatures(degree = deg)

x_poly2 = poly2.fit_transform(x_test)
poly2.fit(x_poly2, y_test)
model.fit(x_poly2, y_test)

x_pred3 = model.predict(x_poly2)

r2c = r2_score(y_test, x_pred3)
print("R2 Score for Testing Data: ", r2c)'''


# In[27]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

unwanted_columns= ['dteday', 'instant', 'casual', 'registered']
df = df.drop(unwanted_columns, axis=1)

continuos_columns = ['temp', 'atemp', 'hum', 'windspeed']
categorical_columns = ['season', 'yr', 'hr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

X = df.drop('cnt', axis=1)
y=df['cnt']

scaler = MinMaxScaler()
df[continuos_columns]=scaler.fit_transform(X[continuos_columns])
X=pd.get_dummies(X, columns=categorical_columns)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 70)
X.head()


# In[30]:


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
x_transformed_poly = poly.fit_transform(X)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(x_transformed_poly,y, test_size = 0.2, random_state = 42)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_poly)

y_train_pred_poly = poly_model.predict(X_train_poly)
y_test_pred_poly = poly_model.predict(X_test_poly)

r2_train_poly = r2_score(y_train_poly, y_train_pred_poly)

print("R2 for Train: ", r2_train_poly)


r2_test_poly = r2_score(y_test_poly, y_test_pred_poly)

print("R2 for test: ", r2_test_poly)


# ## Regularization

# In[31]:


from sklearn.linear_model import Ridge, Lasso, ElasticNet

poly = PolynomialFeatures(degree=2, include_bias =False)

x_train_poly = poly.fit_transform(X_train)
x_test_poly = poly.fit_transform(X_test)

model = Lasso()
model.fit(X_train_poly, y_train)

y_pred =model.predict(X_train_poly)
R2_train = r2_score(y_train, y_pred)

print("R2 train: ", R2_train)
y_pred = model.predict(X_test_poly)
R2_test = r2_score(y_test, y_pred)

print("R2 test: ", R2_test)


# In[33]:


from sklearn.linear_model import Ridge, Lasso, ElasticNet

poly = PolynomialFeatures(degree=2, include_bias =False)

x_train_poly = poly.fit_transform(X_train)
x_test_poly = poly.fit_transform(X_test)

model = Ridge()
model.fit(X_train_poly, y_train)

y_pred =model.predict(X_train_poly)
R2_train = r2_score(y_train, y_pred)

print("R2 train: ", R2_train)
y_pred = model.predict(X_test_poly)
R2_test = r2_score(y_test, y_pred)

print("R2 test: ", R2_test)


# In[32]:


from sklearn.linear_model import Ridge, Lasso, ElasticNet

poly = PolynomialFeatures(degree=2, include_bias =False)

x_train_poly = poly.fit_transform(X_train)
x_test_poly = poly.fit_transform(X_test)

model = ElasticNet(alpha=0.1)
model.fit(X_train_poly, y_train)

y_pred =model.predict(X_train_poly)
R2_train = r2_score(y_train, y_pred)

print("R2 train: ", R2_train)
y_pred = model.predict(X_test_poly)
R2_test = r2_score(y_test, y_pred)

print("R2 test: ", R2_test)


# In[ ]:




