import pandas as pd
import numpy as np

# Load data

url = "https://datascience.quantecon.org/assets/data/wdi_data.csv"
df = pd.read_csv(url)

# df.to_csv("./output_data/wdi.csv")
print(df.info())

print(df.head())

df_small = df.head(5)
print(df_small)

df_tiny = df.iloc[[0, 3, 2, 4], :]
print(df_tiny)

im_ex = df_small[["Imports", "Exports"]]
im_ex_copy = im_ex.copy()
print(im_ex_copy)

print(im_ex + im_ex_copy)

im_ex_tiny = df_tiny + im_ex
print(im_ex_tiny)

# Setting the index

df_year = df.set_index(["year"])
print(df_year.head())

# data fro specific year

data_2010 = df_year.loc[2010]
print(data_2010)

# Compute difference in the average of all variables from one year to another

diff = df_year.loc[2009].mean() - df_year.loc[2008].mean()
print(diff)

# Computing net exports and investment from the WDI dataset

# GDP=Consumption+Investment+GovExpend+NetExports
# Investment=GDP−Consumption−GovExpend−NetExports

# Setting a Hierarchical index

# To achieve multiple columns in the index, we pass a list of multiple column names to set_index.
wdi = df.set_index(["country", "year"])
print(wdi.head(25))

# Slicing a hierarchical index

us_gdp = wdi.loc[("United States", 2010), "GDP"]
print(us_gdp)

uk_germany_gdp = wdi.loc[(["United Kingdom", "Germany"], 2010), "GDP"]
print(uk_germany_gdp)

##Note

# list in row slicing will be an “or” operation, where it chooses rows based
# on whether the index value corresponds to any element of the list.

# tuple in row slicing will be used to denote a single hierarchical index
# and must include a value for each level.

print(wdi.loc["United States"])
print(wdi.loc[("United States", 2010)])
print(wdi.loc[["United States", "Canada"]])
print(wdi.loc[(["United States", "Canada"], [2010, 2011]), :])
print(wdi.loc[[("United States", 2010), ("Canada", 2011)], :])

# pd.IndexSlice

print(wdi.loc[pd.IndexSlice[:, [2005, 2007, 2009]], :])

# Multi-index columns

wdiT = wdi.T
print(wdiT)

print(wdiT.loc[:, "United States"])
print(wdiT.loc[:, ["United States", "Canada"]])
print(wdiT.loc[:, (["United States", "Canada"], 2010)])
