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
