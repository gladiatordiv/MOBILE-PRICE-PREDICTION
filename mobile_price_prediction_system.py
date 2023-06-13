#!/usr/bin/env python
# coding: utf-8

# ### <font color = 'black'> Import Modules </font>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error


# ## Loading the Dataset 

# In[2]:


df=pd.read_csv("ndtv_dataset.csv")
df.head()


# ## Getting Info of the Dataset

# In[3]:


df.info()


# ### <font color = 'black'> Read and Prep File </font>

# In[4]:


df=pd.read_csv("ndtv_dataset.csv")
df_copy=df.copy()

df=df.astype({'Front camera':'object','Number of SIMs':'object','Processor':'object','Rear camera':'object'})
df.drop(['F1','3G', '4g/ Lte', 'Bluetooth', 'GPS', 'Touchscreen', 'Wi-Fi','Resolution', 'Resolution x', 'Resolution y','Battery capacity (mAh) (bin)'],axis=1,inplace=True)
df=pd.get_dummies(df,drop_first=True)


# ### <font color ='k'> Train Test Split </font>

# In[5]:


X=df.drop('Price',axis=1)
y=df['Price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=45)


# ### <font colour = 'black'> Linear Regression </font>

# In[6]:


reg=LinearRegression()
reg.fit(X_train,y_train)
trainpred=reg.predict(X_train)
testpred=reg.predict(X_test)
trainscore=reg.score(X_train,y_train)
testscore=reg.score(X_test,y_test)


# In[7]:


print("Train score :", trainscore)
print("Test score : ", testscore)


# ### <font color = 'k'>Ridge </font>

# In[8]:


Q=[i for i in range(1,1000)]
for i in Q:
    rr = Ridge(alpha=i)
    model = rr.fit(X_train,y_train)
    tr_pred = model.predict(X_train)
    ts_pred = model.predict(X_test)
    ts_score = rr.score(X_test,y_test)
    tr_score = rr.score(X_train,y_train)
    print("alpha",i,"\ttr_Score",round(tr_score,4),"\tts_Score",round(ts_score,4))


# ### <font color='k'> Lasso </font>

# In[9]:


Q=[i for i in range(1,1000)]
for i in Q:
    ls = Lasso(alpha=i)
    model = ls.fit(X_train,y_train)
    tr_pred = model.predict(X_train)
    ts_pred = model.predict(X_test)
    ts_score = ls.score(X_test,y_test)
    tr_score = ls.score(X_train,y_train)
    print("alpha",i,"\ttr_Score",round(tr_score,4),"\tts_Score",round(ts_score,4))


# ### <font color='k'> Cross Validation </font>

# In[10]:


Q=[(1+i/10) for i in range(1,1000)]
tuning_grid = {"alpha":Q}
ls = Ridge()

from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(ls,tuning_grid,scoring="neg_mean_squared_error",cv=4)

cvmodel = cv.fit(X,y)
cvmodel.best_params_


# In[11]:


tuning_grid = {"alpha":[3.1]}
ls = Ridge()

cv = GridSearchCV(ls,tuning_grid,scoring="neg_mean_squared_error",cv=4)

cvmodel = cv.fit(X,y)
cv.predict(X)

