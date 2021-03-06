from matplotlib.pyplot import contour
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


# Optimization in consumer theory


def U(A, B, alpha=1 / 3):
    return B ** alpha * A ** (1 - alpha)


fig, ax = plt.subplots()
B = 1.5
A = np.linspace(1, 10, 100)
ax.plot(A, U(A, B))
ax.set_xlabel("A")
ax.set_ylabel("B")


# change in utility with different bundles

fig, ax = plt.subplots()
B = np.linspace(1, 20, 100).reshape((100, 1))
print(B.flatten())
contours = ax.contourf(A, B.flatten(), U(A, B))
fig.colorbar(contours)

ax.set_xlabel("A")
ax.set_ylabel("B")
ax.set_title("U(A,B)")

# consumer indifference


def A_indifference(B, ubar, alpha=1 / 3):
    return ubar ** (1 / (1 - alpha)) * B ** (-alpha / (1 - alpha))


def plot_indifference_curves(ax, alpha=1 / 3):
    ubar = np.arange(1, 11, 2)
    ax.plot(B, A_indifference(B, ubar, alpha))
    ax.legend([r"$\bar{U}$" + " = {}".format(i) for i in ubar])
    ax.set_xlabel("B")
    ax.set_ylabel(r"$A(B, \bar{U}$)")


fig, ax = plt.subplots()
plot_indifference_curves(ax)


# Consumer budget constraint


def A_bc(B, W=20, pa=2):
    """Given B, W and pa return the max amount of A a consumer can affors"""
    return (W - B) / pa


def plot_budget_constraint(ax, W=20, pa=2):
    B_bc = np.array([0, W])
    A = A_bc(B_bc, W, pa)
    ax.plot(B_bc, A)
    ax.fill_between(B_bc, 0, A, alpha=0.2)
    ax.set_xlabel("B")
    ax.set_ylabel("A")
    return ax


fig, ax = plt.subplots()
plot_budget_constraint(ax, 20, 2)


#Optimal choice

fig, ax = plt.subplots()
plot_indifference_curves(ax)
plot_budget_constraint(ax)

plt.show()