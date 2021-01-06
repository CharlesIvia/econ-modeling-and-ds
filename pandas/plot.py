import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as transforms
import quandl
import qeds

qeds.themes.mpl_style()

quandl.ApiConfig.api_key = os.environ.get(
    "QUANDL_AUTH", "Dn6BtVoBhzuKTuyo6hbp")

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

fig, ax = plt.subplots(1, 2, figsize=(10, 6))


ax[0].plot(aapl[["Adj. Low", "Adj. High"]])
ax[1].plot(aapl[["Low", "High"]])

plt.show()
