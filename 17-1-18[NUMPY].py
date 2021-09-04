#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

an_array = np.array([12,45,25])
print(an_array)


# In[4]:


print(an_array.shape)


# In[5]:


print(type(an_array))


# In[7]:


another_array = np.array([[1,2,4],[3,2,1]])
print(another_array)


# In[8]:


print(another_array.shape)


# In[14]:


ex1=np.zeros((2,2))
print(ex1)


# In[16]:


ex2 = np.full((2,2),9.0)
print(ex2)


# In[17]:


ex3 = np.random.random((2,2))
print(ex3)


# In[18]:


a_array = np.array([[22,32,13,78],[24,56,45,55],[34,66,55,89]])
print(a_array)


# In[19]:


a_slice = a_array[:2,1:3]
print(a_slice)


# # Boolean Indexing

# In[20]:


b_array = np.array([[12,34,55],[32,53,43]])
print(b_array)


# In[22]:


filter =(b_array>20)
filter


# In[23]:


print(b_array[filter])


# In[24]:


b_array[b_array>15]     #another method.


# In[25]:


b_array[(b_array>20)&(b_array<40)]


# ## ndarray DATATYPES & Operation

# In[26]:


ab = np.array([11,12])
print(ab.dtype)


# In[27]:


ab = np.array([11,12] ,dtype=np.int64)
print(ab.dtype)


# In[28]:


ba = np.array([56.90,65.97] ,dtype=np.int64)    #and vice-versa with float
print(ba.dtype)
print()
print(ba)


# In[29]:


x=np.array([[12,31],[32,12]] , dtype=np.int64)
y=np.array([[42,11],[54,76]] , dtype=np.float64)
print(x)
print()
print(y)


# In[30]:


print(x+y)
print()                     #similarly subtract,mul,div etc
print(np.add(x,y))


# In[31]:


print(np.sqrt(x))


# In[32]:


print(np.exp(y))    #expotential e**y


# ## Statistical , sorting & Set
# 

# In[33]:


print(np.mean(x))


# In[34]:


print(np.mean(y),axis=1)


# In[39]:


print(x.mean())


# In[36]:


print(x.mean(axis=1))  #row=1 col=0


# In[38]:


print(x.sum())


# In[43]:


print(np.median(x,axis=0))


# In[44]:


unsorted = np.random.random(10)
print(unsorted)


# In[47]:


sorted = np.array(unsorted)   #copy and sort
sorted.sort()
print(sorted)


# In[49]:


unsorted.sort()   #direct sorting
print(unsorted)


# In[51]:


w = np.array([2,3,4,1,2,3,2])
print(np.unique(w))


# In[52]:


s1 =np.array([["chair","desk","table"],["fan","table","hut"]])
s2=np.array([["bed","stool","desk"],["cat","dog","hut"]])
print(s1,s2)


# In[60]:


print(np.intersectnd(s1, s2))


# ### Broadcasting

# In[63]:


start = np.zeros((4,3))
print(start)


# In[64]:


add_rows = np.array([1,0,2])
print(add_rows)


# In[65]:


z= start + add_rows
print(z)


# In[70]:


add_cols=np.array([0,1,2,3])
add_cols = add_cols.T
print(add_cols)


# In[69]:


v = add_cols + start
print(v)


# In[ ]:




