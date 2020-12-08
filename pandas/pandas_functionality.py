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
