#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 


# In[7]:



d=pd.read_csv('http://bit.ly/w-data')
d.head()


# In[3]:


#rows&colums
d.shape


# In[4]:


d['Hours'].unique()


# In[5]:


d['Scores'].unique()


# In[6]:


#check_null
d.isnull().sum()


# In[8]:


d.describe()


# # visualisation

# In[11]:


plt.scatter(x='Hours',y='Scores',data=d)
plt.xlabel='hours'
plt.ylabel='Score'
plt.title('Hours vs Score')


# In[14]:


sns.regplot(x='Hours',y='Scores',data=d)


# In[73]:


X=d[['Hours']]
Y=d[['Scores']]
from sklearn.model_selection import train_test_split 

X_train,X_test,Y_train,Y_test=train_test_split(X, Y,test_size=0.2, random_state=0)  
print(X_train.shape,X_test.shape,Y_test.shape,X_train.shape)


# In[93]:


X_train


# In[92]:


Y_test


# # LINEAR_REGRESSION

# In[ ]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()


# In[91]:


#model_Train
lm.fit(X_train,Y_train)


# In[79]:


Yhat=lm.predict(X_test)
Yhat


# In[84]:


#RMSerror
np.mean((Y_test-Yhat)**2)


# # Model visualisation

# In[85]:


aa=sns.distplot(d['Scores'],hist=False,color='r',label='Actual value')
ap=sns.distplot(Yhat,hist=False,color='b',label='predic value')


# In[32]:


lm.score(X_test,Y_test)


# In[88]:


hour=[[9.25]]
predicted_score=lm.predict(hour)
predicted_score


# In[ ]:




