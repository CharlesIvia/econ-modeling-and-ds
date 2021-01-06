import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as transforms
import quandl
import qeds

qeds.themes.mpl_style()

quandl.ApiConfig.api_key = os.environ.get("QUANDL_AUTH", "Dn6BtVoBhzuKTuyo6hbp")

# WANT: Visualize the impact of iphone announcements on apple share prices

# First: Create a series containing the date of each iPhone announcement

announcement_dates = pd.Series(
    [
        "First iPhone",
        "3G",
        "3GS",
        "4",
        "4S",
        "5",
        "5S/5C",
        "6/6 Plus",
        "6S/6S Plus",
        "7/7 Plus",
        "8/8 Plus/X",
    ],
    index=pd.to_datetime(
        [
            "Jan. 9, 2007",
            "Jun. 9, 2008",
            "Jun. 8, 2009",
            "Jan. 11, 2011",
            "Oct. 4, 2011",
            "Sep. 12, 2012",
            "Sep. 10, 2013",
            "Sep. 9, 2014",
            "Sep. 9, 2015",
            "Sep. 7, 2016",
            "Sep. 12, 2017",
        ]
    ),
    name="Model",
)

print(announcement_dates)

# Second: Grab apple stock price from quandl

aapl = quandl.get("WIKI/AAPL", start_date="2006-12-25")

print(aapl.head())

# Create subplots

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

# plot the Adjusted open to account for stock split

ax1.plot(aapl["Adj. Open"])

# get the figure so we can re-display the plot after making changes
# fig = ax.get_figure()

# Set the title
ax1.set_title("AAPL Adjusted opening price")

# Other customizations
ax1.set_ylim(0, 200)
ax1.set_yticks([0, 50, 100, 150, 200])

# Other plots

ax2.plot(aapl[["Adj. Low", "Adj. High"]])

plt.show()
