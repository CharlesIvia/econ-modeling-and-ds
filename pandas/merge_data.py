import pandas as pd
import qeds
import matplotlib.pyplot as plt

# activate plot thene

qeds.themes.mpl_style()

# Data used in this script

# WDI data on GDP components, population, and square miles of countries
# Book ratings: 6,000,000 ratings for the 10,000 most rated books on Goodreads
# Details for all delayed US domestic flights in November 2016, obtained from the Bureau of Transportation Statistics


## from WDI. Units trillions of 2010 USD

url = "https://datascience.quantecon.org/assets/data/wdi_data.csv"
wdi = pd.read_csv(url).set_index(["country", "year"])

print(wdi.info())
print(wdi)

wdi2017 = wdi.xs(2017, level="year")
print(wdi2017)

wdi2016_17 = wdi.loc[pd.IndexSlice[:, [2016, 2017]], :]
print(wdi2016_17)

# Data from https://www.nationmaster.com/country-info/stats/Geography/Land-area/Square-miles
# units -- millions of square miles

sq_miles = pd.Series(
    {
        "United States": 3.8,
        "Canada": 3.8,
        "Germany": 0.137,
        "United Kingdom": 0.0936,
        "Russia": 6.6,
    },
    name="sq_miles",
).to_frame()

sq_miles.index.name = "country"

print(sq_miles)

# from WDI. Units millions of people

# from WDI. Units millions of people
pop_url = "https://datascience.quantecon.org/assets/data/wdi_population.csv"
pop = pd.read_csv(pop_url).set_index(["country", "year"])
print(pop.info())
print(pop.head(10))

## Suppose that we were asked to compute a number of statistics with the data above:

# As a measure of land usage or productivity, what is Consumption per square mile?
# What is GDP per capita (per person) for each country in each year? How about Consumption per person?
# What is the population density of each country? How much does it change over time?

##Notice we have to combine data from more than one dataset to answer above questions

# pd.concat([Dfs], axis)

# axis = 0, stacks DFs on top of one another

print(pd.concat([wdi2017, sq_miles], axis=0))

# axis = 1, stacks data side by side

print(pd.concat([wdi2017, sq_miles], axis=1))

# Want: Determine what is the consumption per square mile?

temp = pd.concat([wdi2017, sq_miles], axis=1)
print(temp["Consumption"] / temp["sq_miles"])


# pd.merge

# pd.merge operates on two DataFrames at a time and is primarily used to bring
# columns from one DataFrame into another, aligning data based on one or more “key” columns


print(pd.merge(wdi2017, sq_miles, on="country"))

print(pd.merge(wdi2016_17, sq_miles, on="country"))

print(pd.merge(wdi2016_17.reset_index(), sq_miles, on="country"))

# Multiple columns

print(pd.merge(wdi2016_17, pop, on=["country", "year"]))

# Want: Determine capita income and consumption per person for each country and year

wdi_pop = wdi_pop = pd.merge(wdi2016_17, pop, on=["country", "year"])

print(wdi_pop["GDP"] / wdi_pop["Population"])
print(wdi_pop["Consumption"] / wdi_pop["Population"])

# Optional arguments for merge include - on, left_on, right_on

# left_on and right_on are useful when DFs have same name for a particular column

# left_index and right_index

# how

wdi2017_no_US = wdi2017.drop("United States")
print(wdi2017_no_US)

sq_miles_no_germany = sq_miles.drop("Germany")
print(sq_miles_no_germany)

# default
print(pd.merge(wdi2017, sq_miles, on="country", how="left"))

# notice ``Russia`` is included
print(pd.merge(wdi2017, sq_miles, on="country", how="right"))

# notice no United States or Russia
print(pd.merge(wdi2017_no_US, sq_miles, on="country", how="inner"))

# includes all 5, even though they don't all appear in either DataFrame
print(pd.merge(wdi2017_no_US, sq_miles_no_germany, on="country", how="outer"))

# df.merge(df2)
# Note that the DataFrame type has a merge method.

# It is the same as the function we have been working with, but passes the DataFrame before the period as left.

# Thus df.merge(other) is equivalent to pd.merge(df, other).

print(wdi2017.merge(sq_miles, on="country", how="right"))

# df.join
# The join method for a DataFrame is very similar to the merge method described above, but
# only allows you to use the index of the right DataFrame as the join key.
