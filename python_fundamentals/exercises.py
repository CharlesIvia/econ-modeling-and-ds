import math
from math import exp

# Verify the "trick" where the percent difference (x−y/x)
# between two numbers close to 1 can be well approximated by the difference between the log of the two numbers (log(x)−log(y))

x = 1.05
y = 1.02


def pec_diff(x, y):
    return ((x - y) / x) * 100


print(pec_diff(x, y))


def log_xy(x, y):
    return (math.log(x) - math.log(y)) * 100


print(log_xy(x, y))


# replace

test = "abc"
print(test.replace("c", "d"))

# String formating

# Iterations

students = ["Alpha", "Harriet", "Joy", "Kamau"]
marks = [72, 65, 66, 76]

en = enumerate(students)
print(list(en))

for index, student in enumerate(students):
    mark = marks[index]
    print(f"{student} has {mark} points")


# Comprehension

# List comprehension

xs = list(range(10))

x_comp = [x ** 2 for x in xs]
print(x_comp)

# Create a dictionary from lists
tickers = ["AAPL", "GOOGL", "TVIX"]
prices = [175.96, 1047.43, 8.38]

z = zip(tickers, prices)
print(list(z))

d = {key: value for key, value in zip(tickers, prices)}

print(d)

# Create a list from a dictionary
d = {"AMZN": "Seattle", "TVIX": "Zurich", "AAPL": "Cupertino"}

hq = [d[ticker] for ticker in d.keys()]
print(hq)

# Create a nested dictionary

gdp_data = [9.607, 10.48, 11.06]
years = [2013, 2014, 2015]
exports = [
    {"manufacturing": 2.4, "agriculture": 1.5, "services": 0.5},
    {"manufacturing": 2.5, "agriculture": 1.4, "services": 0.9},
    {"manufacturing": 2.7, "agriculture": 1.4, "services": 1.5},
]

data = zip(years, gdp_data, exports)

data_dict = {year: {"gdp": gdp, "exports": export} for year, gdp, export in data}
print(data_dict)

# total exports of services by year

expo_by_year = [data_dict[year]["exports"]["services"] for year in data_dict.keys()]
print(expo_by_year)
