#!/usr/bin/env python
#import numpy
import numpy as np

an_array = np.array([12,45,25])
print(an_array)

print(an_array.shape)
print(type(an_array))


another_array = np.array([[1,2,4],[3,2,1]])
print(another_array)

print(another_array.shape)

ex1=np.zeros((2,2))
print(ex1)

ex2 = np.full((2,2),9.0)
print(ex2)

#array using random number
ex3 = np.random.random((2,2))
print(ex3)


a_array = np.array([[22,32,13,78],[24,56,45,55],[34,66,55,89]])
print(a_array)


a_slice = a_array[:2,1:3]
print(a_slice)


# # Boolean Indexing

b_array = np.array([[12,34,55],[32,53,43]])
print(b_array)

#filtering based on specific condition
filter =(b_array>20)
filter

print(b_array[filter])

#another method.
b_array[b_array>15]     

b_array[(b_array>20)&(b_array<40)]


#ndarray DATATYPES & Operation

ab = np.array([11,12])
print(ab.dtype)

ab = np.array([11,12] ,dtype=np.int64)
print(ab.dtype)

#with float
ba = np.array([56.90,65.97] ,dtype=np.int64)   
print(ba.dtype)
print()
print(ba)

x=np.array([[12,31],[32,12]] , dtype=np.int64)
y=np.array([[42,11],[54,76]] , dtype=np.float64)
print(x)
print()
print(y)

print(x+y)
print()   

#Mathematical function
print(np.add(x,y))
print(np.sqrt(x))
print(np.exp(y))   

# ## Statistical , sorting & Set

print(np.mean(x))
print(np.mean(y),axis=1)
print(x.mean())

print(x.mean(axis=1))  #row=1 col=0
print(x.sum())
print(np.median(x,axis=0))

unsorted = np.random.random(10)
print(unsorted)
#copy and sort
sorted = np.array(unsorted)   
sorted.sort()
print(sorted)

#direct sorting
unsorted.sort()   
print(unsorted)

w = np.array([2,3,4,1,2,3,2])
print(np.unique(w))


s1 =np.array([["chair","desk","table"],["fan","table","hut"]])
s2=np.array([["bed","stool","desk"],["cat","dog","hut"]])
print(s1,s2)

print(np.intersectnd(s1, s2))

# ### Broadcasting
start = np.zeros((4,3))
print(start)
add_rows = np.array([1,0,2])
print(add_rows)
z= start + add_rows
print(z)

add_cols=np.array([0,1,2,3])
add_cols = add_cols.T
print(add_cols)

v = add_cols + start
print(v)
