#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
# import os # accessing directory structure
import pandas as pd 
import seaborn as sns


# In[2]:


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.processing import StandardScales
# from mpl_toolkits.mplot3d import Axes3D
# import os


# In[3]:


tt= pd.read_csv("titanic_train.csv") 


# In[4]:


tt.head()


# In[5]:


tt.describe()


# In[6]:


tt.shape


# In[7]:


tt.dtypes


# In[8]:


tt.columns.values


# In[9]:


tt.info()


# In[10]:


tt.isnull().sum()


# In[11]:


tt.drop(columns=["Cabin"],inplace=True)


# In[12]:


tt.isnull().sum()


# In[13]:


tt.shape


# In[14]:


tt['Age'].fillna(tt['Age'].mean(),inplace=True)


# In[15]:


tt.isnull().sum()


# In[16]:


plt.figure(figsize=(8,6))
sns.heatmap(tt.isnull())


# In[17]:


sns.heatmap(tt.isnull(),yticklabels=False,cbar=False,cmap="plasma")


# In[18]:


tt.duplicated().sum()


# In[19]:


sns.boxplot(data=tt)


# In[20]:


tt.dtypes


# In[21]:


sns.boxplot(data=tt, x= "Age")


# In[22]:


sns.boxplot(data=tt, x= "PassengerId")


# In[23]:


sns.boxplot(data=tt, x= "Fare")   # dot after range describes outlayers


# # Univarite analysis

# In[24]:


sns.displot(data=tt, x="Age") #histogram  


# In[25]:


sns.displot(data=tt, x="Age", kde=True) #histogram


# In[26]:


sns.displot(data=tt, x="Age", kind="kde") 


# In[27]:


sns.displot(data=tt, x="Fare", ) #histogram


# In[28]:


sns.countplot(data=tt, x="Sex", ) 


# In[29]:


sns.countplot(data=tt, x="Sex", hue="Survived") 


# In[30]:


sns.countplot(data=tt, x="Embarked",) 


# In[ ]:




