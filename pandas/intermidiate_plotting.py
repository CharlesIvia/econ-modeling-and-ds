import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as transforms
import quandl
import qeds
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

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

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

# plot the Adjusted open to account for stock split
# Remember the independent variable goes into y axis and dependent variable into x axis

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
ax3.plot(aapl[["Low", "High"]])

# Data Cleaning

bday_us = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def neigbor_dates(date, nbefore=3, nafter=3):
    # Make sure the date is a datetime
    date = pd.to_datetime(date)

    # Create a list of business days

    before_and_after = [date + i * bday_us for i in range(-nbefore,
                                                          nafter + 1)]
    return before_and_after


dates = []

for ann_date in announcement_dates.index:
    dates.extend(neigbor_dates(ann_date))
dates = pd.Series(dates)

# Index into our DataFrrame using the new dates

prices = aapl.loc[dates]
print(prices.head())

# Bring information on iPhone models into the DataFrame

prices = prices.join(announcement_dates)
print(prices["Model"].isnull().sum())
print(prices["Model"].head(7))

# Fill the NaNs below with Data

prices = prices.ffill(limit=3)
print(prices["Model"].isnull().sum())
print(prices["Model"].head(7))

# Fill NaNs above with Data

prices = prices.bfill(limit=3)
print(prices["Model"].isnull().sum())
print(prices["Model"].head(7))

# Normalize the share price on each announcement date to 100 and scale neigbors accordingly


def scale_by_middle(df):
    # How many rows
    N = df.shape[0]

    # Divide by middle row and scale to 100
    # Note: N // 2 is modulus division meaning that it is
    #       rounded to nearest whole number)
    out = (df["Open"] / df.iloc[N // 2]["Open"]) * 100

    # We don't want to keep actual dates, but rather the number
    # of days before or after the announcment. Let's set that
    # as the index. Note the +1 because range excludes upper limit
    out.index = list(range(-(N // 2), N // 2 + 1))

    # also change the name of this series
    out.name = "DeltaDays"
    return out


to_plot = prices.groupby("Model").apply(scale_by_middle).T

print(to_plot)


#Constructing the plot

#colors

background = tuple(np.array([253, 238, 222]) / 255)
blue = tuple(np.array([20, 64, 134]) / 255)
pink = tuple(np.array([232, 75, 126]) / 255)


def get_color(x):
    if "S" in x:
        return pink
    else:
        return blue

colors = announcement_dates.map(get_color).values

# yticks
yticks = [90, 95, 100, 105, 110, 115]

# plt.show()
