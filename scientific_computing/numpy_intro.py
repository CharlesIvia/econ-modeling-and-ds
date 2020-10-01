# numpy => numerical python

import numpy as np

# create an array from a list

x_id = np.array([1, 2, 3])
print(type(x_id))

print(x_id)

#indexing an array

print(x_id[0])
print(x_id[0:2])

#2-dimensional array(a matrix)

x_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x_2d)

#acessing values/indexing into dimensions - specify row then column

print(x_2d[1, 2])
print(x_2d[0, 0])

#get first and second rows

print(x_2d[0, :])
print(x_2d[1, :])

#get the columns

print(x_2d[:, 0])
print(x_2d[:, 1])

#3-d array

x_3d_list = [[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]]
x_3d = np.array(x_3d_list)
print(x_3d)

print(x_3d[0])
print(x_3d[1])

print(x_3d[0, 1, 0])
print(x_3d[0, 0, :])


#ARRAY FUNCTIONALITY

#array properties- most common is shaape and dtype

x = np.array([[1, 2, 3], [4, 5, 6]])
print(x)
print(x.shape)
print(x.dtype)

rows, columns = x.shape
print(f"rows = {rows}, columns ={columns} ")

x = np.array([True, False, True])
print(x.shape)  # Note that in the above, the (3,) represents a tuple of length 1, distinct from a scalar integer 3.
print(x.dtype)

x = np.array([
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
])

print(x)
print(x.shape)
print(x.dtype)

#creating arrays - np.ones and np.ones

sizes = (2, 3, 4)
x = np.zeros(sizes)
print(x)

y = np.ones((4))
print(y)

#broadcasting operations

#Two types of operations that will be useful for arrays of any dimension are:

# Operations between an array and a single number.
# Operations between two arrays of the same shape.

# Using np.ones to create an array

x = np.ones((2, 2))

print("x = ", x)

print("2 + x = ", 2 + x)

print("2 - x = ", 2 - x)

print("2 * x = ", 2 * x)

print("x / 2 = ", x / 2)

#ops btn two arrays od same size


x = np.array([[1.0, 2.0], [3.0, 4.0]])
y = np.ones((2, 2))


#verify size
print(x.shape)
print(y.shape)

print("x = ", x)

print("y = ", y)

print("x + y = ", x + y)

print("x - y", x - y)

print("(elementwise) x * y = ", x * y)

print("(elementwise) x / y = ", x / y)

#Universal functions

#Below, we will create an array that contains 10 points between 0 and 25.
# This is similar to range -- but spits out 50 evenly spaced points from 0.5
# to 25.

x = np.linspace(0.5, 25, 10)
print(x)

# Applies the sin function to each element of x
print(np.sin(x))


# Takes log of each element of x
print(np.log(x))

#other arrays ops

x = np.linspace(0, 25, 10)
print(np.mean((x)))

print(np.std(x))

# np.min, np.median, etc... are also defined
print(np.max(x))

print(np.diff(x))

#Note that many of these operations can be called as methods on x:

print(x.mean())
print(x.std())
print(x.min())
# print(x.diff())  # this one is not a method...
print(x.reshape((5, 2)))


#EXERCISES

#indexing into a 3d array

j_3d = np.zeros((2, 3, 4))
print(j_3d)


k_3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(k_3d)

print(k_3d[0, :])  # first section of array
print(k_3d[0, 0, :])  # first row
print(k_3d[0, 0, 0])  # first element

print(k_3d[:,:, 0])
