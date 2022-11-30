#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors Classification Model

# AIM: To perform K Nearest Neighbors Classification Model working. 

# Some information related to the K Nearest Neighbors Classification Model. 
# 1. Select the number of neighbors. 
# 2. Based on the number of neighbors, select that many closest labeled dta point, w.r.t the unlabelled point
# 3. Make the closest labeled points to vote 
# 4. Based on the majority assign a label to the unlabelled data

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[40]:


df=pd.read_csv("C:/Users/USER/Desktop/Machine Learnng/Ml_bi_data/Iris.csv")


# In[41]:


df.head()


# In[42]:


df.shape


# In[43]:


df.corr()


# In[44]:


df.describe()


# In[45]:


df["Species"].value_counts()


# In[46]:


df.isnull().sum()


# In[47]:


target=df["Species"]


# In[48]:


feature=df.drop("Species",axis='columns')


# In[49]:


target


# In[50]:


feature


# In[51]:


#here id is not usefull to predict the data so I'm droping the feature column to make best predication. 


# In[52]:


feature=feature.drop("Id",axis="columns")


# In[53]:


feature


# In[54]:


type(feature)


# In[55]:


feature.dtypes


# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


x_train,x_test,y_train,y_test=train_test_split(feature,target,test_size=0.2,random_state=3,stratify=target)


# Here I'm performing the Scalling because to make predication best

# In[58]:


from sklearn.preprocessing import MinMaxScaler


# In[59]:


feature.describe()


# In[60]:


scaler=MinMaxScaler()


# In[61]:


x_train=scaler.fit_transform(x_train)


# In[62]:


x_test=scaler.transform(x_test)


# In[63]:


pd.DataFrame(x_train,columns=feature.columns).describe()


# selecting the model and performing 

# In[64]:


from sklearn.neighbors import KNeighborsClassifier


# In[66]:


model=KNeighborsClassifier(n_neighbors=9,metric="euclidean")


# In[67]:


model.fit(x_train,y_train)


# In[68]:


model.score(x_test,y_test)


# this 0.9 is no.of correct prediction / total no.of prediction 

# In[70]:


feature


# In[75]:


data=pd.DataFrame({"SepalLengthCm":[5.1],"SepalWidthCm":[3.5],"PetalLengthCm":[1.4],"PetalWidthCm":[0.2]})


# In[76]:


data


# In[77]:


data=scaler.transform(data)


# In[78]:


model.predict(data)


# In[ ]:





# In[ ]:




