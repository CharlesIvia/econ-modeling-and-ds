#visualizing data using matplotlib

import matplotlib.pyplot as plt
import numpy as np
import qeds
# qeds.themes.mpl_style();


#first plot

#step 1- create a figure and axis object which stores info from our graph

fig, ax = plt.subplots()

#step 2 - generate data that we plot

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

#step 3 - make a plot on our axis
ax.plot(x, y)

#difference btw figure and axis

#axis is the canvas where we draw our plots
#figure is the entire framed painting which incliudes axis itself

fig, ax = plt.subplots()

fig.set_facecolor("yellow")
ax.set_facecolor("grey")


# We specified the shape of the axes -- It means we will have two rows and three columns
# of axes on our figure
fig, axes = plt.subplots(2, 3)
fig.set_facecolor("yellow")


# Can choose hex colors
colors = ["#065535", "#89ecda", "#ffd1dc", "#ff0000", "#6897bb", "#9400d3"]


# axes is a numpy array and we want to iterate over a flat version of it
for (ax, c) in zip(axes.flat, colors):
    ax.set_facecolor(c)

fig.tight_layout()


#Functionality

#BAR

countries = ["CAN", "MEX", "USA"]
populations = [36.7, 129.2, 325.700]
land_area = [3.850, 0.761, 3.790]

fig, ax = plt.subplots(2)

ax[0].bar(countries, populations, align="center")
ax[0].set_title("Populations (in millions)")

ax[1].bar(countries, land_area, align="center")
ax[1].set_title("Land area (in millions miles squared)")


fig.tight_layout()


#SCATTER AND ANNOTATION

N = 50

np.random.seed(42)
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)

area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

fig, ax = plt.subplots()
ax.scatter(x, y, s=area, c=colors, alpha=0.5)

ax.annotate(
    "First point", xy=(x[0], y[0]), xycoords="data",
    xytext=(25, -25), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.6")
)


#FILL BETWEEN

x = np.linspace(0, 1, 500)
y = np.sin(4 * np.pi * x) * np.exp(-5 * x)

fig, ax = plt.subplots()

ax.grid(True)
ax.fill(x, y)

plt.show()