# Differencing is a method of transforming a time series dataset.
# Can be used to remove the series dependence on time, ie.temporal dependence.
# This includes structures like trends and seasonality.

# Lag difference:
# Taking the difference between consecutive observations is called a lag-1 difference
# For time series with a seasonal component, the lag may be expected to be the period (width) of the seasonality

# Difference Order
# The process of differencing can be repeated more than once until all temporal dependence has been removed.
# The number of times that differencing is performed is called the difference order

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from pandas.core.series import Series

# Read data

data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"


def parser(x):
    return datetime.strptime("190" + x, "%Y-%m")


def plot_line(ser):
    ser.plot()
    plt.show()


series = pd.read_csv(
    data_url, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser
)

print(series.head())
plot_line(series)

# create a differenced series


def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


X = series.values
diff = difference(X)
print(diff)
plot_line(diff)

# automatic differencing - diff()
auto_diff = series.diff()
print(auto_diff)
plot_line(auto_diff)
