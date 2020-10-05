# import numpy to prepare for code below
import numpy as np
import matplotlib.pyplot as plt

# activate plot theme
import qeds
qeds.themes.mpl_style();

#Elementwise operations

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print("Element-wise Addition", x + y)
print("Element-wise Subtraction", x - y)
print("Element-wise Multiplication", x * y)
print("Element-wise Division", x / y)

#Scalar operations

print("Scalar Addition", 3 + x)
print("Scalar Subtraction", 3 - x)
print("Scalar Multiplication", 3 * x)
print("Scalar Division", 3 / x)

#Dot product

print("Dot product with @ ", x@y)

#MATRICES

x = np.array([[1, 2, 3], [4, 5, 6]])
print(x)

y = np.ones((2, 3))
print(y)

z = np.array([[1, 2], [3, 4], [5, 6]])
print(z)

#Elelemnt-wise and scalar operations on matricess

print("Element-wise Addition\n", x + y)
print("Element-wise Subtraction\n", x - y)
print("Element-wise Multiplication\n", x * y)
print("Element-wise Division\n", x / y)


print("Scalar Addition\n", 3 + x)
print("Scalar Subtraction\n", 3 - x)
print("Scalar Multiplication\n", 3 * x)
print("Scalar Division\n", 3 / x)

#Matrix multiplication

x1 = np.reshape(np.arange(6), (3, 2))
x2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
x3 = np.array([[2, 5, 2], [1, 2, 1]])
x4 = np.ones((2, 3))

y1 = np.array([1, 2, 3])
y2 = np.array([0.5, 0.5])

print(x1)
print(x2)
print(x3)
print(x4)
print(y1)
print(y2)

print("Using the matmul function for two matrices")
print(np.matmul(x1, x4))

print("Using the dot function for two matrices")
print(np.dot(x1, x4))

print("Using @ for two matrices")
print(x1 @ x4)

print("Using the matmul function for vec and mat")
print(np.matmul(y1, x1))
print("Using the dot function for vec and mat")
print(np.dot(y1, x1))
print("Using @ for vec and mat")
print(y1 @ x1)

#Matrix transpose

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("x transpose is")
print(x.transpose())

#Identity matrix -When we multiply any matrix or vector
# by the identity matrix, we get the original matrix or vector back!

I = np.eye(3)
print(I)

x = np.reshape(np.arange(9), (3, 3))
print(x)
y = np.array([1, 2, 3])
print(y)

print("I @ x", "\n", I @ x)
print("x @ I", "\n", x @ I)
print("I @ y", "\n", I @ y)
print("y @ I", "\n", y @ I)