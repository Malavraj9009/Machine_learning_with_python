#!/usr/bin/env python
# coding: utf-8

# ##LOAN PREADICTION PROBLEM AV PRACTICE DATASET

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib as plt 


# In[2]:


df = pd.read_csv("C:/Users/malav/Desktop/SE/AV/Loan Prediction/train_AV_prac.csv") #Reading the dataset in a dataframe using Pandas


# ##DaTA EXPLORATAION

# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df['Property_Area'].value_counts() ##Non-numerical frequency table


# In[6]:


df['ApplicantIncome'].hist(bins=50)


# In[7]:


df.boxplot(column='ApplicantIncome')


# In[8]:


df.boxplot(column='ApplicantIncome', by = 'Education')


# In[9]:


df.boxplot(column='LoanAmount')


# In[35]:


import seaborn as sns 


# In[36]:


sns.boxplot(df['LoanAmount']) 


# ##DATA CLEANING AND PREPROCESSING STEPS

# In[12]:


temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean()) 
print('Frequency Table for Credit History:') 
print(temp1) 
print('\nProbility of getting loan for each Credit History class:')
print (temp2)


# In[15]:


import matplotlib.pyplot as plt


# In[18]:


fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History") 
temp1.plot(kind='bar') 
ax2 = fig.add_subplot(122)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
temp2.plot(kind = 'bar')


# In[19]:


temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[20]:


df.apply(lambda x: sum(x.isnull()),axis=0) ##CHECKING NULL VALUES


# In[21]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True) #FILLING NULLVALUES WITH MEAN


# In[22]:


df['Self_Employed'].fillna('No',inplace=True)  #filling NA with No as 86% of them have no


# In[24]:


table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median) 


# In[25]:


# Define function to return value of this pivot_table 
def fage(x): 
    return table.loc[x['Self_Employed'],x['Education']] 


# In[27]:


# Replace missing values
df['LoanAmount']
[df['LoanAmount'].isnull()] = df[df['LoanAmount'].isnull()].apply(fage, axis=1)


# In[28]:


##HERE WE ARE TRANSFORMING THE DATA SO THAT THE OUTLIERS ARE RETAINED AND NOT REMOVED
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[29]:


##CREATING NEW FEATURE FROM 2 EXISTING FEATURES
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 


# In[32]:


#creating another feature i.e payback so that we can know how the customer can pay back
df['Payback']=df['LoanAmount']/df['TotalIncome']
df['Payback']


# In[ ]:





# In[ ]:





# In[ ]:




