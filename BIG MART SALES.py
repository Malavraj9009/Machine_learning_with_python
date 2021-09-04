#!/usr/bin/env python
# coding: utf-8


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,roc_auc_score


#Reading the file
BMS=pd.read_csv('/train_Bi_mart_sales.csv')


#doing some basic EDA
BMS.head(5)
type(BMS)
BMS.describe()

 ##weight and size has missing values so we will impute them with mean
pd.isna(BMS).sum()
BMS['Item_Weight'].fillna(BMS['Item_Weight'].mean(), inplace=True)

#FILLING NULLVALUES WITH MEAN

pd.isna(BMS).sum()
str(BMS['Outlet_Size'])

 ##checking datatypes
BMS.dtypes               
type(BMS['Outlet_Size'])

#filling NA with Medium as mode is Medium
BMS['Outlet_Size'].fillna('Medium',inplace=True)  

pd.isna(BMS).sum()

#using LabelEncoder
le=LabelEncoder()


#BMS['Item_Identifier']=le.fit_transform(BMS['Item_Identifier'])
#BMS['Outlet_Identifier']=le.fit_transform(BMS['Outlet_Identifier'])
BMS['Item_Fat_Content']=le.fit_transform(BMS['Item_Fat_Content'])
BMS['Item_Type']=le.fit_transform(BMS['Item_Type'])
BMS['Outlet_Size']=le.fit_transform(BMS['Outlet_Size'])
BMS['Outlet_Location_Type']=le.fit_transform(BMS['Outlet_Location_Type'])
BMS['Outlet_Type']=le.fit_transform(BMS['Outlet_Type'])

BMS.head(4)

#BMS_encoded=BMS.apply(le.fit_transform)
#BMS_encoded=BMS_encoded.iloc[:,[1,2,3,4,5,7,8,9,10]] ##selecting only relevant columns


#Training testing splitting
target=BMS.Item_Outlet_Sales
BMS = BMS.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'], axis=1)
BMS.head(5)
target.shape
BMS.shape

X_train,X_test,Y_train,Y_test=train_test_split(BMS,target,test_size=0.2)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#BMS = BMS.apply(pd.to_numeric, errors='coerce')
#target = target.apply(pd.to_numeric, errors='coerce')


#Making model

#Creating Random Forest
Rand_regressor=RandomForestRegressor(n_estimators=1000,criterion='mse', random_state=0, n_jobs=-1)
Rand_regressor.fit(X_train,Y_train)
print(BMS.columns,Rand_regressor.feature_importances_)

#From above table we can see Item_visiblity,Item_MRP,Outlet_type are imp and make almost 70-80%
sfm = SelectFromModel(Rand_regressor, threshold=0.10)
# Train the selector
sfm.fit(X_train, Y_train)
plt.plot(sfm)
#y_pred=Rand_regressor.predict(X_test)
Rand_regressor.score(X_test,Y_test)

# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)


# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
X_important_train.shape
Y_train.shape

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, Y_train)
#sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
#sel.fit(X_train, Y_train)
#this is for classifier
# To see which features are important we can use get_support method on the fitted model.
sel.get_support()

#We can now make a list and count the selected features.
selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)

#Creating Linear Regression and fitting it
lin_reg=LinearRegression()

model=lin_reg.fit(X_train,Y_train)
predictions=lin_reg.predict(X_test)

## The line / model
plt.scatter(Y_cv, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")


# Model Accuracy
mean_absolute_error(predictions,Y_test)
)##root mean squared error
np.sqrt(mean_squared_error(predictions,Y_test
                           
#BMS_decoded=BMS_encoded.apply(le.inverse_transform)


# # WORKING ON SEPARATE TEST DATA
BMS_test=pd.read_csv('/Practice_problems/test_BMSI.csv')

BMS_test.head(4)
pd.isna(BMS_test).sum()
BMS_test['Item_Weight'].fillna(BMS_test['Item_Weight'].mean(), inplace=True)
                           
#FILLING NULLVALUES WITH MEAN
BMS_test['Outlet_Size'].fillna('Medium',inplace=True)
pd.isna(BMS_test).sum()

#BMS_test['Item_Identifier']=le.fit_transform(BMS_test['Item_Identifier'])
#BMS_test['Outlet_Identifier']=le.fit_transform(BMS_test['Outlet_Identifier'])
BMS_test['Item_Fat_Content']=le.fit_transform(BMS_test['Item_Fat_Content'])
BMS_test['Item_Type']=le.fit_transform(BMS_test['Item_Type'])
BMS_test['Outlet_Size']=le.fit_transform(BMS_test['Outlet_Size'])
BMS_test['Outlet_Location_Type']=le.fit_transform(BMS_test['Outlet_Location_Type'])
BMS_test['Outlet_Type']=le.fit_transform(BMS_test['Outlet_Type'])

BMS_test.head(4)
BMS_test.shape
BMS_test.dtypes
BMS_test_1 = BMS_test.drop(['Item_Identifier','Outlet_Identifier'], axis=1)
BMS_test.head(4)

BMS_test_1.head(5)
predictions_test=lin_reg.predict(BMS_test)
predictions_test
BMS_test['Item_Outlet_Sale']=predictions_test

BMS_test['Item_Identifier']=le.transform(BMS_test['Item_Identifier'])
BMS_test['Outlet_Identifier']=le.transform(BMS_test['Outlet_Identifier'])


#random forest regressor

print(BMS_test)
Pred_test=Rand_regressor.predict(BMS_test_1)
BMS_test_1['Item_Outlet_Sales']=Pred_test
#Rand_regressor.score(Y_TESTING,BMS_TESTING)

#Converting the df to csv  
BMS_test_1.to_csv('/Practice_problems/submission_2.csv')

