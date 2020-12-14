import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# State unemployment data

# Download data directly from a url and read it into pandas DF

url = "https://datascience.quantecon.org/assets/data/state_unemployment.csv"
unemp_raw = pd.read_csv(url, parse_dates=["Date"])
print(unemp_raw.head())

# Goal - look at unemployment rate across different states over time

unemp_all = unemp_raw.reset_index().pivot_table(
    index="Date", columns="state", values="UnemploymentRate"
)

print(unemp_all.head())

states = [
    "Arizona",
    "California",
    "Florida",
    "Illinois",
    "Michigan",
    "New York",
    "Texas",
]

unemp = unemp_all[states]
print(unemp.head())

unemp.plot()

# Dates in pandas

print(unemp.index)

# Data corresponding to a single date

print(unemp.loc["01/01/2000", :])

# Data for all days between New Years Day and June first in the year 2000

print(unemp.loc["01/01/2000":"06/01/2000", :])

# DataFrame Aggregations

# Built-in Aggregations

print(unemp.mean())

# Default aggregation is by column
# Can use axis keyword argument to aggregate by row

print(unemp.mean(axis=1).head())
print(unemp.var(axis=1).head())

# Writing Own Aggregations

# Write a Python function that takes a Series as an input and outputs a single value.
# Call the agg method with our new function as an argument.

# Classifying employment status of states

#
# Step 1: We write the (aggregation) function that we'd like to use
#


def high_or_low(s):
    """
    This function takes a pandas Series object and returns high
    if the mean is above 6.5 and low if the mean is below 6.5
    """
    if s.mean() < 6.5:
        out = "Low"
    else:
        out = "High"
    return out


#
# Step 2: Apply it via the agg method.
#

print(unemp.agg(high_or_low))

# agg can take multiple fns at once

print(unemp.agg([min, max, high_or_low]))

# Transforms - many analytical operations do not necessarily involve an aggregation

# Built in Transforms

print(unemp.head())

print(unemp.pct_change().head())
print(unemp.diff().head())

# Transforms are either series transforms or scalar transforms

# Custom series transforms

##Steps
# Write a Python function that takes a Series and outputs a new Series.
# Pass our new function as an argument to the apply method
# (alternatively,the transform method).

# Standardize unemployment data

#
# Step 1: We write the Series transform function that we'd like to use
#


def standardize_data(x):
    """
    Changes the data in a Series to become mean 0 with standard deviation 1
    """
    mu = x.mean()
    std = x.std()

    return (x - mu) / std


#
# Step 2: Apply our function via the apply method.
#
std_unemp = unemp.apply(standardize_data)
print(std_unemp.head())

# Takes the absolute value of all elements of a function
abs_std_unemp = std_unemp.abs()

print(abs_std_unemp.head())

# find the date when unemployment was "most different from normal" for each State


def idxmax(x):
    # idxmax of Series will return index of maximal value
    return x.idxmax()


print(abs_std_unemp.agg(idxmax))

# Boolean selection

unemp_small = unemp.head()
print(unemp_small)

# Using a list of booleans to select

print(unemp_small.loc[[True, True, True, False, False]])

# Add a second argument to select column, ":" means all

print(unemp_small.loc[[True, False, True, False, True], :])

# Add booleans to select both rows and columns
print(
    unemp_small.loc[
        [True, True, True, False, False], [True, False, False, False, False, True, True]
    ]
)

# Creating boolean dataframes/series

unemp_texas = unemp_small["Texas"] < 4.5
print(unemp_texas)

# extract subset of rows from a df

print(unemp_small.loc[unemp_texas])

unemp_ny_vs_texas = unemp_small["New York"] > unemp_small["Texas"]
print(unemp_ny_vs_texas)

print(unemp_small.loc[unemp_ny_vs_texas])

# multiple conditions

# (bool_series1) & (bool_series2)
# (bool_series1) | (bool_series2)

small_NYTX = (unemp_small["Texas"] < 4.7) & (unemp_small["New York"] < 4.7)
print(small_NYTX)
print(unemp_small[small_NYTX])

# isin() instead of |

print(unemp_small["Michigan"].isin([3.3, 3.2]))

# select full rows where this Series is True
print(unemp_small.loc[unemp_small["Michigan"].isin([3.3, 3.2])])


# .any() and  .all()
# any returns True whenever at least one of the inputs are True
# while all is True only when all the inputs are True.

# Want: Count the number of months in which all states in our sample had unemployment above 6.5%

# construct the DataFrame of bools
high = unemp > 6.5
print(high.head())

# use the .all method on axis=1 to get the dates where all states have a True
all_high = high.all(axis=1)
print(all_high.head())

print(all_high.sum())

# Call .sum to add up the number of True values in `all_high`
# (note that True == 1 and False == 0 in Python, so .sum will count Trues)

msg = "Out of {} months, {} had high unemployment accros all states"
print(msg.format(len(all_high), all_high.sum()))
