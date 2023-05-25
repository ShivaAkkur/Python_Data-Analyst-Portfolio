#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("D:/Business Analytics/Mall_Customers.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.tail()


# # Univariate Analysis

# In[6]:


df.describe()


# In[7]:


sbn.distplot(df['Annual Income (k$)']);


# In[8]:


sbn.displot(df['Annual Income (k$)']);


# In[9]:


sbn.histplot(df['Annual Income (k$)']);


# In[10]:


df.columns


# In[11]:


columns=['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sbn.distplot(df[i])


# In[12]:


sbn.kdeplot(df['Annual Income (k$)'],shade=True);


# In[13]:


sbn.kdeplot(df['Annual Income (k$)'], shade=True, hue=df['Gender'])


# In[14]:


sbn.kdeplot(df['Annual Income (k$)'], shade=False, hue=df['Gender'])


# In[15]:


df.columns


# In[16]:


columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sbn.kdeplot(df[i],shade=True,hue=df['Gender']);


# In[17]:


columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sbn.boxplot(data=df, x='Gender', y=df[i]);


# In[18]:


columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sbn.distplot(df[i]);


# In[19]:


import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


# In[20]:


df=pd.read_csv("D:\Business Analytics\Mall_Customers.csv")


# In[21]:


df.head


# In[22]:


df.head()


# In[23]:


df['Gender']. value_counts()


# In[24]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[25]:


df.columns


# In[26]:


sbn.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')


# In[27]:


df=df.drop('CustomerID', axis=1)
sbn.pairplot(data=df)


# In[28]:



sbn.pairplot(df, hue='Gender')


# In[29]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']. mean()


# In[30]:


df.corr()


# In[31]:


sbn.heatmap(df.corr(),annot=True,cmap='coolwarm')


# # Clustering-Univariate, Bivariate, Multivariate

# In[32]:


clustering1 = KMeans(n_clusters = 3)


# In[33]:


clustering1.fit(df[['Annual Income (k$)']])


# In[34]:


clustering1.fit(df[['Annual Income (k$)']])


# In[35]:


clustering1.labels_


# In[36]:


df['Income Cluster']=clustering1.labels_
df.head()


# In[37]:


df


# In[38]:


df['Income Cluster'].value_counts()


# In[39]:


clustering1.inertia_


# In[40]:


inertia_scores=[]
for i in range(1,11):
    Kmeans=KMeans(n_clusters=i)
    Kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(Kmeans.inertia_)


# In[41]:


inertia_scores


# In[42]:


plt.plot(range(1,11),inertia_scores)


# In[43]:


df.columns


# In[44]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)']. mean()


# In[45]:


df.groupby('Income Cluster')['Age','Annual Income (k$)','Spending Score (1-100)'].median()


# In[46]:


df.groupby('Income Cluster')['Age','Annual Income (k$)','Spending Score (1-100)'].std()


# # Bivariate Clustering

# In[47]:


clustering2=KMeans()
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Spending and Income Cluster']=clustering2.labels_
df.head()


# In[48]:


inertia_scores2=[]
for i in range(1,11):
    Kmeans2=KMeans(n_clusters=i)
    Kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores.append(Kmeans2.inertia_)


# In[49]:


sbn.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')


# In[50]:


df.head()


# In[51]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#import warnings
#warnings.filterwarnings("ignore")


# In[52]:


df=pd.read_csv("D:\Business Analytics\Mall_Customers.csv")


# In[53]:


df.describe


# In[54]:


df.head()


# # Bivariate Clustering

# In[55]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# In[56]:


inertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),inertia_scores2)


# In[62]:


centers=pd.DataFrame(clustering2.cluster_centers_)
centers.columns=['x','y']


# In[95]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue= 'Spending and Income Cluster', palette='tab10')
plt.savefig('clustering_bivariate.png') # to save visual


# In[65]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[71]:


df.groupby('Spending and Income Cluster')['Age','Annual Income (k$)','Spending Score (1-100)'].mean()


# # Multivariate Clustering

# In[72]:


from sklearn.preprocessing import StandardScaler


# In[73]:


scale = StandardScaler


# In[74]:


df.head()


# In[78]:


dff=pd.get_dummies(df,drop_first=True) # to drop a column Gender_Female we used drop_first=True


# In[79]:


dff.head()


# In[80]:


dff.columns


# In[83]:


dff=dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff.head()


# In[96]:


dff


# In[97]:


dff.to_csv('clustering.csv') # to save dff file


# # Analysis

# #Target Cluster

# In[ ]:


# Target would be cluster1 which has have high spending score and high income
# 54% of cluster1 shoppers are women. we should look for ways to attract these customers using a marketing campaign targeting popular items in this cluster.
# cluster2 presents an interesting opportunity to market to the customers for sales event on popular items.

