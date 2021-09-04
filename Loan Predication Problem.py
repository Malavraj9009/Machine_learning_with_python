#!/usr/bin/env python
# coding: utf-8

# ##LOAN PREADICTION PROBLEM 

#Import libraries
import pandas as pd
import numpy as np 
import matplotlib as plt 
import seaborn as sns 

#Reading the dataset in a dataframe using Pandas
df = pd.read_csv("./train_AV_prac.csv")


#DaTA EXPLORATAION
df.head()
df.describe()
df['Property_Area'].value_counts() ##Non-numerical frequency table
df['ApplicantIncome'].hist(bins=50)
df.boxplot(column='ApplicantIncome')
df.boxplot(column='ApplicantIncome', by = 'Education')
df.boxplot(column='LoanAmount')

sns.boxplot(df['LoanAmount']) 


#DATA CLEANING AND PREPROCESSING STEPS

temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean()) 
print('Frequency Table for Credit History:') 
print(temp1) 
print('\nProbility of getting loan for each Credit History class:')
print (temp2)



#Plotting graphs
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

temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

##CHECKING NULL VALUES
df.apply(lambda x: sum(x.isnull()),axis=0) 
#FILLING NULLVALUES WITH MEAN
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True) 
#filling NA with No as 86% of them have no
df['Self_Employed'].fillna('No',inplace=True)  
table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median) 

# Define function to return value of this pivot_table 
def fage(x): 
    return table.loc[x['Self_Employed'],x['Education']] 

# Replace missing values
df['LoanAmount']
[df['LoanAmount'].isnull()] = df[df['LoanAmount'].isnull()].apply(fage, axis=1)


##TRANSFORMING THE DATA SO THAT THE OUTLIERS ARE RETAINED AND NOT REMOVED
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


##CREATING NEW FEATURE FROM 2 EXISTING FEATURES
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 

#creating another feature i.e payback so that we can know how the customer can pay back
df['Payback']=df['LoanAmount']/df['TotalIncome']
df['Payback']

