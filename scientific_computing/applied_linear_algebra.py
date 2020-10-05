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