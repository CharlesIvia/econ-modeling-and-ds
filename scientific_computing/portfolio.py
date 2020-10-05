import numpy as np

#Static payoffs - As an example, consider a portfolio with 4 units of asset A, 2.5 units of asset B, and 8 units of asset C.

#At a particular point in time, the assets pay 3/unit of asset A, 5/unit of B, and 1.10/unit of C.

port_value = 4 * 3 + 2.5 * 5 + 8 * 1.1
print(port_value)

x = np.array([4, 2.5, 8])  # portfolio units
y = np.array([3, 5, 1.1])  # payoffs

n = len(x)
p = 0

for i in range(n):
    p = p + x[i] * y[i]

print(p)

print(np.dot(x, y))


