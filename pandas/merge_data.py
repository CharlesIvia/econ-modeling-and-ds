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
