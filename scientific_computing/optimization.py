import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import ptp
import scipy.optimize as opt


# derivatives and optima


def f(x):
    return x ** 4 - 3 * x ** 2


def fp(x):
    return 4 * x ** 3 - 6 * x


# Create 100 evenly spaced points between -2 and 2

x = np.linspace(-2, 2, 100)

# Evaluate the functions at x values

fx = f(x)
fpx = fp(x)

# Create a plot

fig, ax = plt.subplots(1, 2)

ax[0].plot(x, fx)
ax[0].set_title("Function")


ax[1].plot(x, fpx)
ax[1].hlines(0.0, -2.5, 2.5, color="k", linestyle="--")
ax[1].set_title("Derivative")

# For a scalar problem, we give it the function and the bounds between
# which we want to search
neg_min = opt.minimize_scalar(f, [-2, -0.5])
pos_min = opt.minimize_scalar(f, [0.5, 2.0])
print("The negative minimum is: \n", neg_min)
print("The positive minimum is: \n", pos_min)


plt.show()
