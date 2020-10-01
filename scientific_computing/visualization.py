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

plt.show()
