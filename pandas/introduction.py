import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SERIES - a single column of data

# A Series containing US unemployment rates since 1995 to 2015

values = [5.6, 5.3, 4.3, 4.2, 5.8, 5.3, 4.6, 7.8, 9.1, 8.0, 5.7]

years = list(range(1995, 2017, 2))

unemp = pd.Series(data=values, index=years, name="US Unemployment")

print(unemp)

indexes = unemp.index
print(indexes)

# .head and .tail

print(unemp.head())
print(unemp.tail())

# Basic plotting

unemp.plot()
# plt.show()

# Unique values

unique = unemp.unique()
print(unique)

# Indexing .loc[index_items]

unemp_1995 = unemp.loc[1995]
print(unemp_1995)

# DATAFRAMES - how pandas stores one+ columns of data

data = {
    "NorthEast": [5.9, 5.6, 4.4, 3.8, 5.8, 4.9, 4.3, 7.1, 8.3, 7.9, 5.7],
    "MidWest": [4.5, 4.3, 3.6, 4.0, 5.7, 5.7, 4.9, 8.1, 8.7, 7.4, 5.1],
    "South": [5.3, 5.2, 4.2, 4.0, 5.7, 5.2, 4.3, 7.6, 9.1, 7.4, 5.5],
    "West": [6.6, 6.0, 5.2, 4.6, 6.5, 5.5, 4.5, 8.6, 10.7, 8.5, 6.1],
    "National": [5.6, 5.3, 4.3, 4.2, 5.8, 5.3, 4.6, 7.8, 9.1, 8.0, 5.7],
}

unemp_region = pd.DataFrame(data, index=years)
print(unemp_region)

print(unemp_region.index)
print(unemp_region.values)

# .head and .tail in a dataframe

print(unemp_region.head())
print(unemp_region.tail(3))

unemp_region.plot()
# plt.show()

# Indexing a dataframe .loc[row, column]

print(unemp_region.loc[1995, "NorthEast"])
print(unemp_region.loc[[1995, 2005], "South"])
print(unemp_region.loc[1995, ["NorthEast", "National"]])
print(unemp_region.loc[:, "NorthEast"])
print(unemp_region["NorthEast"])

# Computations with Columns

# Divide by 100 to move from percent units to a rate
print(unemp_region["West"] / 100)
print(unemp_region["West"].max())

# Difference between two columns

print(unemp_region["West"] - unemp_region["MidWest"])

# Find correlation between two columns

print(unemp_region.West.corr(unemp_region["MidWest"]))

# find correlation between all column pairs

print(unemp_region.corr())

# Data Types

# Booleans (bool)
# Floating point numbers (float64)
# Integers (int64)
# Dates (datetime)
# Categorical data (categorical)
# Everything else, including strings (object)

str_unemp = unemp_region.copy()
str_unemp["South"] = str_unemp["South"].astype(str)
print(str_unemp.dtypes)

print(str_unemp.head())  # Looks okay but ...

print(str_unemp.sum())

# Changing dataframes

# New columns - df["New Column Name"] = new_values

unemp_region["UnweightedMean"] = (
    unemp_region["NorthEast"]
    + unemp_region["MidWest"]
    + unemp_region["South"]
    + unemp_region["West"]
) / 4

print(unemp_region.head())

# Changing values  df.loc[index, column] = value

unemp_region.loc[1995, "UnweightedMean"] = 0.0
print(unemp_region.head())

# Rename columns

names = {"NorthEast": "NE", "MidWest": "MW", "South": "S", "West": "W"}

unemp_region.rename(columns=names)
print(unemp_region.head())

unemp_region_shortname = unemp_region.rename(columns=names)
print(unemp_region_shortname)
