from numpy.core.fromnumeric import choose
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from pandas.core.tools import numeric
import qeds

df = pd.DataFrame(
    {
        "numbers": ["#23", "#24", "#18", "#14", "#12", "#10", "#35"],
        "nums": ["23", "24", "18", "14", np.nan, "XYZ", "35"],
        "colors": ["green", "red", "yellow", "orange", "purple", "blue", "pink"],
        "other_column": [0, 1, 0, 2, 1, 0, 2],
    }
)

print(df)

# this generates an error - coul not convert ... to numeric

# print(df["numbers"].mean())

# STRING METHODS

# int(replace("#", ""))

# Slow method - iterating over all rows

for row in df.iterrows():
    index_value, column_values = row

    clean_number = int(column_values["numbers"].replace("#", " "))
    df.at[index_value, "numbers_loop"] = clean_number

print(df["numbers_loop"].mean())

# Faster method - apply a sting method to an entire column of data
# s.str.method_name

df["numbers_str"] = df["numbers"].str.replace("#", " ")
print(df)

# We can use .str to access almost any string method that works on normal strings

print(df["colors"].str.contains("p"))
print(df["colors"].str.capitalize())

# TYPE CONVERSIONS

print(df["numbers_str"].dtype)

# convert to numbers using pd.to_numeric

df["numbers_numeric"] = pd.to_numeric(df["numbers_str"])
print(df["numbers_numeric"].dtype)

# using astype() we can convert to any supported stype

df["numbers_numeric"].astype(str)
df["numbers_numeric"].astype(float)

# MISSING DATA

print(df)

# find missing data using isnull method

print(df.isnull())

# We might want to know whether particular rows or columns have any missing data.

# To do this we can use the .any method on the boolean DataFrame df.isnull().

print(df.isnull().any(axis=0))
print(df.isnull().any(axis=1))

# Approaches to missing data

# 1. Exclusion: Ignore any data that is missing (.dropna)
# 2. Imputation: Compute “predicted” values for the data that is missing (.fillna)

# drop all rows containing a missing observation
print(df.dropna())

# fill the missing values with a specific value
print(df.fillna(value=100))

# use the _next_ valid observation to fill the missing data
print(df.fillna(method="bfill"))

# use the _previous_ valid observation to fill missing data
print(df.fillna(method="ffill"))

# Case Study: a dataset of 2000 Chipotle orders

chiptole = qeds.data.load("chipotle_raw")
print(chiptole.head())

# Want: Find out what is the average price of an item with chicken?

chiptole_price = chiptole["item_price"].str.replace("$", " ")
chiptole["numeric_price"] = pd.to_numeric(chiptole_price)

print(chiptole.head())

chiptole["chicken_dish"] = chiptole["item_name"].str.contains("Chicken" or "chicken")
print(chiptole.head(10))

contains_chicken = chiptole["chicken_dish"] == True
chicken_df = chiptole.loc[contains_chicken]
print(chicken_df)

average_chicken_dish_price = chicken_df["numeric_price"].mean()
print(average_chicken_dish_price)

# The average price of an item with chicken is $10.13

# Want: Find out what is the average price of an item with steak?

chiptole["steak_dish"] = chiptole["item_name"].str.contains("Steak" or "steak")
print(chiptole.head(10))

contains_steak = chiptole["steak_dish"] == True
steak_df = chiptole.loc[contains_steak]
print(steak_df)

average_steak_dish_price = steak_df["numeric_price"].mean()
print(average_steak_dish_price)

# The average price of an item with steak at is $10.52

##Want: Find out which items produced more revenue between steak and chicken

print(
    "Chicken dishes produced ${} in revenue.".format(chicken_df["numeric_price"].sum())
)
print("Steak dishes produced ${} in revenue.".format(steak_df["numeric_price"].sum()))


##Want: Determine number of missing items in the dataset and in columns

print(chiptole.isnull().value_counts())
print(chiptole.isnull().sum())
