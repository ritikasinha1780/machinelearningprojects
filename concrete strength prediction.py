#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# In[2]:


concrete = pd.read_excel("Concrete_Data.xls")


# In[3]:


get_ipython().run_line_magic('pip', 'install xlrd')


# In[4]:


concrete = pd.read_excel("Concrete_Data.xls")


# In[5]:


concrete.head()


# In[6]:


concrete.columns = ['cement','blastFurnace','flyAsh','water','superplasticizer','courseAggregate','fineaggregate','age','strength']


# In[7]:


concrete


# In[8]:


concrete.shape


# In[9]:


concrete.isnull().sum()


# In[10]:


concrete.duplicated().sum()


# In[11]:


concrete.drop_duplicates()


# In[12]:


concrete.info()


# In[13]:


concrete.describe()


# In[14]:


corr = concrete.corr()
corr


# In[15]:


sns.heatmap(corr,annot=True,cbar=True, cmap='coolwarm')


# In[16]:


from sklearn.model_selection import train_test_split
X = concrete.drop("strength", axis=1)
y = concrete["strength"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


for col in X_train.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)
 
    
    
#     # QQ plot
    plt.subplot(122)
    stats.probplot(X_train[col], dist='norm',plot=plt)
    plt.title(col)
    plt.show()


# In[18]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox') 


# In[19]:


X_train_transformed = pt.fit_transform(X_train+0.000001)
X_test_transoformed = pt.transform(X_test+0.000001)


# In[20]:


X_train_transformed = pd.DataFrame(X_train_transformed, columns=X_train.columns)

for col in X_train_transformed.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)
 
    
    
#     # QQ plot
    plt.subplot(122)
    sns.distplot(X_train_transformed[col])
    plt.title(col)
    plt.show()


# In[21]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training set only.
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train_transformed = scaler.transform(X_train)
X_test_transoformed = scaler.transform(X_test)


# In[23]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import r2_score


# Define the regression models
models = {
    'lin_reg': LinearRegression(),
    'ridge': Ridge(),
    'lasso': Lasso(),   
}




for name ,model in models.items():
    model.fit(X_train_transformed,y_train)
    y_pred = model.predict(X_test_transoformed)
    
    print(f"{name} : {r2_score(y_test,y_pred)}")


# In[29]:


ridge = Ridge()
ridge.fit(X_train_transformed,y_train)
y_pred = ridge.predict(X_test_transoformed)
r2_score(y_test,y_pred)


# In[30]:


def predicion_system(cem,blastf,flyas,water,superplaster,courseagg,fineagg,age):
    features = np.array([[cem,blastf,flyas,water,superplaster,courseagg,fineagg,age]])
    prediction = ridge.predict(features).reshape(1,-1)
    
    return prediction[0]


# In[31]:


X_train


# In[32]:


cem = 158.60
blastf = 148.90
flyas = 116.00
water = 175.10
superplaster = 15.00
courseagg = 953.3
fineagg = 719.70
age = 28

prediction = predicion_system(cem,blastf,flyas,water,superplaster,courseagg,fineagg,age)
print("strength is : ",prediction)


# In[ ]:




