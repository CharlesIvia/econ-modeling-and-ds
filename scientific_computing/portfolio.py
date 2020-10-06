import numpy as np
import matplotlib.pyplot as plt

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

#Pricing different portfolios


y = np.array([3.0, 5.0, 1.1])  # payoffs
x1 = np.array([4.0, 2.5, 8.0])  # portfolio 1
x2 = np.array([2.0, 1.5, 0.0])  # portfolio 2

X = np.array((x1, x2))
print(X)

p1 = np.dot(X[0, :], y)
print(p1)

print(X[0, :])

#NPV OF A PORTFOLIO

#Depreciation of production rates

gamma_A = 0.8
gamma_B = 0.9

#Interest rate discounting

r = 0.05

discount = np.array([(1 / (1+r)) ** t for t in range(20)])
print(discount)

#Create arrays with production of each oilfield

oil_A = 5 * np.array([gamma_A**t for t in range(20)])
oil_B = 2 * np.array([gamma_B**t for t in range(20)])

oil_fields = np.array([oil_A, oil_B])
print(oil_fields)

# Use matrix multiplication to get discounted sum of oilfield values
# and then sum the two values


Vs = oil_fields @ discount
print(Vs)

print(f"The value of oilfields is {Vs.sum()}")


#NPV for a portfolio with infinite lifetime

#How different is this infinite horizon approximation from the T = 20

# Depreciation of production rates
gamma_A = 0.80
gamma_B = 0.90

# Interest rate discounting
r = 0.05


def infhor_NPV_oilfield(stating_output, gamma, r):
    beta = gamma / (1+r)
    return stating_output / (1 - beta)


def compute_NPV_oilfield(starting_output, gamma, r, T):
    outputs = starting_output * np.array([gamma**t for t in range(T)])
    discount = np.array([(1 / (1+r))**t for t in range(T)])
    npv = np.dot(outputs, discount)
    return npv

Ts = np.arange(2, 75)
NPVs_A = np.array([compute_NPV_oilfield(5, gamma_A, r, t) for t in Ts])
NPVs_B = np.array([compute_NPV_oilfield(2, gamma_B, r, t) for t in Ts])

NPVs_T = NPVs_A + NPVs_B
NPV_oo = infhor_NPV_oilfield(
    5, gamma_A, r) + infhor_NPV_oilfield(2, gamma_B, r)
print(NPV_oo)

