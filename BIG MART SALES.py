#!/usr/bin/env python
# coding: utf-8

# # Big mart sales 
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[3]:


BMS=pd.read_csv('C:/Users/malav/Desktop/SE/AV/BIG MART SALES/train_Bi_mart_sales.csv')


# In[4]:


BMS.head(5)


# In[5]:


type(BMS)


# In[6]:


BMS.describe()


# In[7]:


pd.isna(BMS).sum() ##weight and size has missing values so we will impute them with mean


# In[39]:


BMS['Item_Weight'].fillna(BMS['Item_Weight'].mean(), inplace=True)
#FILLING NULLVALUES WITH MEAN


# In[ ]:


pd.isna(BMS).sum()


# In[ ]:


str(BMS['Outlet_Size'])


# In[ ]:


BMS.dtypes                ##checking datatypes


# In[ ]:


type(BMS['Outlet_Size'])


# In[8]:


BMS['Outlet_Size'].fillna('Medium',inplace=True)  #filling NA with Medium as mode is Medium


# In[ ]:


pd.isna(BMS).sum()


# In[9]:


from sklearn.preprocessing import OneHotEncoder,LabelEncoder


# In[10]:


le=LabelEncoder()


# In[11]:


#BMS['Item_Identifier']=le.fit_transform(BMS['Item_Identifier'])
#BMS['Outlet_Identifier']=le.fit_transform(BMS['Outlet_Identifier'])
BMS['Item_Fat_Content']=le.fit_transform(BMS['Item_Fat_Content'])
BMS['Item_Type']=le.fit_transform(BMS['Item_Type'])
BMS['Outlet_Size']=le.fit_transform(BMS['Outlet_Size'])
BMS['Outlet_Location_Type']=le.fit_transform(BMS['Outlet_Location_Type'])
BMS['Outlet_Type']=le.fit_transform(BMS['Outlet_Type'])


# In[12]:


BMS.head(4)


# In[ ]:


#BMS_encoded=BMS.apply(le.fit_transform)


# In[ ]:



#BMS_encoded=BMS_encoded.iloc[:,[1,2,3,4,5,7,8,9,10]] ##selecting only relevant columns


# # Training testing splitting

# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


target=BMS.Item_Outlet_Sales


# In[47]:


BMS = BMS.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'], axis=1)


# In[48]:


BMS.head(5)


# In[49]:


target.shape
BMS.shape


# In[50]:


X_train,X_test,Y_train,Y_test=train_test_split(BMS,target,test_size=0.2)


# In[51]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[ ]:


#BMS = BMS.apply(pd.to_numeric, errors='coerce')
#target = target.apply(pd.to_numeric, errors='coerce')


# # Making model
# 

# In[52]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


# In[ ]:


# Here I will do the model fitting and feature selection altogether in one line of code.


# In[53]:


Rand_regressor=RandomForestRegressor(n_estimators=1000,criterion='mse', random_state=0, n_jobs=-1)


# In[54]:


Rand_regressor.fit(X_train,Y_train)


# In[ ]:


print(BMS.columns,Rand_regressor.feature_importances_)


# In[ ]:


#From above table we can see Item_visiblity,Item_MRP,Outlet_type are imp and make almost 70-80%


# In[57]:


sfm = SelectFromModel(Rand_regressor, threshold=0.10)


# In[58]:


# Train the selector
sfm.fit(X_train, Y_train)


# In[60]:


plt.plot(sfm)


# In[ ]:


#y_pred=Rand_regressor.predict(X_test)


# In[ ]:


##accuracy_score(y_pred,Y_test) THIS IS FOR CLASSIFICATION


# In[56]:


Rand_regressor.score(X_test,Y_test)


# In[ ]:


# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)


# In[ ]:


# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)


# In[ ]:


X_important_train.shape


# In[ ]:


Y_train.shape


# In[ ]:


# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, Y_train)


# In[ ]:


#sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
#sel.fit(X_train, Y_train)
#this is for classifier


# In[ ]:


# To see which features are important we can use get_support method on the fitted model.


# In[ ]:


sel.get_support()


# In[ ]:


#We can now make a list and count the selected features.


# In[ ]:


selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)


# In[ ]:


lin_reg=LinearRegression()


# In[ ]:


model=lin_reg.fit(X_train,Y_train)


# In[ ]:


predictions=lin_reg.predict(X_test)


# In[ ]:


## The line / model
plt.scatter(Y_cv, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")


# Model Accuracy

# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,roc_auc_score


# In[ ]:


mean_absolute_error(predictions,Y_test)


# In[ ]:


np.sqrt(mean_squared_error(predictions,Y_test)##root mean squared error


# In[ ]:


#BMS_decoded=BMS_encoded.apply(le.inverse_transform)


# # WORKING ON SEPARATE TEST DATA

# In[84]:


BMS_test=pd.read_csv('C:/Users/malav/Desktop/SE/AV/Practice_problems/test_BMSI.csv')


# In[ ]:


BMS_test.head(4)


# In[ ]:


pd.isna(BMS_test).sum()


# In[85]:


BMS_test['Item_Weight'].fillna(BMS_test['Item_Weight'].mean(), inplace=True)
#FILLING NULLVALUES WITH MEAN


# In[86]:


BMS_test['Outlet_Size'].fillna('Medium',inplace=True)


# In[77]:


pd.isna(BMS_test).sum()


# In[87]:


#BMS_test['Item_Identifier']=le.fit_transform(BMS_test['Item_Identifier'])
#BMS_test['Outlet_Identifier']=le.fit_transform(BMS_test['Outlet_Identifier'])
BMS_test['Item_Fat_Content']=le.fit_transform(BMS_test['Item_Fat_Content'])
BMS_test['Item_Type']=le.fit_transform(BMS_test['Item_Type'])
BMS_test['Outlet_Size']=le.fit_transform(BMS_test['Outlet_Size'])
BMS_test['Outlet_Location_Type']=le.fit_transform(BMS_test['Outlet_Location_Type'])
BMS_test['Outlet_Type']=le.fit_transform(BMS_test['Outlet_Type'])


# In[88]:


BMS_test.head(4)


# In[89]:


BMS_test.shape


# In[90]:


BMS_test.dtypes


# In[91]:


BMS_test_1 = BMS_test.drop(['Item_Identifier','Outlet_Identifier'], axis=1)


# In[92]:


BMS_test.head(4)


# In[93]:


BMS_test_1.head(5)


# In[ ]:


predictions_test=lin_reg.predict(BMS_test)


# In[ ]:


predictions_test


# In[ ]:


BMS_test['Item_Outlet_Sale']=predictions_test


# In[ ]:


BMS_test['Item_Identifier']=le.transform(BMS_test['Item_Identifier'])
BMS_test['Outlet_Identifier']=le.transform(BMS_test['Outlet_Identifier'])


# #this is for random forest regressor

# In[72]:


print(BMS_test)


# In[110]:


Pred_test=Rand_regressor.predict(BMS_test_1)


# In[112]:


BMS_test_1['Item_Outlet_Sales']=Pred_test


# In[105]:


#Rand_regressor.score(Y_TESTING,BMS_TESTING)


# In[113]:


BMS_test_1.to_csv('C:/Users/malav/Desktop/SE/AV/Practice_problems/submission_2.csv')

