# WANT: Explore SMA, CMA, EMA usinf rainfall and temperature data from Open Data Barcelona

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")

# load temp data

df_temperature = pd.read_csv("./temperature_barcelona.csv")
print(df_temperature.head())

# load rainfall data

df_rainfall = pd.read_csv("./rainfall_barcelona.csv")
print(df_rainfall)

# basic summary of the data frames

df_temperature.info()
df_rainfall.info()

# Calculate yearly values

# set year column as index

df_temperature.set_index("Any", inplace=True)

df_rainfall.set_index("Any", inplace=True)

# transalte index name into English

df_temperature.index.name = "year"
df_rainfall.index.name = "year"

# Calulate the yeraly average temp and rainfall

df_temperature["average_temperature"] = df_temperature.mean(axis=1)
print(df_temperature)

df_rainfall["accumulated_rainfall"] = df_rainfall.mean(axis=1)
print(df_rainfall)

# drop monthly values

df_temperature = df_temperature[["average_temperature"]]
print(df_temperature)

df_rainfall = df_rainfall[["accumulated_rainfall"]]
print(df_rainfall)

# Visualizing the time series data

# Temperature

# line plot the yearly average temperature in barcelona

df_temperature.plot(color="green", linewidth=2, figsize=(12, 6))

# modify ticks size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend("")

# title and labels
plt.title("The yearly average air temperature in Barcelona", fontsize=20)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Temperature [°C]", fontsize=16)

# Rainfall

# line plot - the yearly accumulated rainfall in Barcelona
df_rainfall.plot(color="steelblue", linewidth=3, figsize=(12, 6))

# modify ticks size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend("")

# title and labels
plt.title("The yearly accumulated rainfall in Barcelona", fontsize=20)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Rainfall [mm]", fontsize=16)


# The Simple Moving Average

# temperature
# the simple moving average over a period of 10 years
df_temperature["SMA_10"] = df_temperature.average_temperature.rolling(
    10, min_periods=1
).mean()

# the simple moving average over a period of 20 year
df_temperature["SMA_20"] = df_temperature.average_temperature.rolling(
    20, min_periods=1
).mean()

print(df_temperature)

# rainfall
# the simple moving average over a period of 10 years
df_rainfall["SMA_10"] = df_rainfall.accumulated_rainfall.rolling(
    10, min_periods=1
).mean()

# the simple moving average over a period of 20 year
df_rainfall["SMA_20"] = df_rainfall.accumulated_rainfall.rolling(
    20, min_periods=1
).mean()

print(df_rainfall)

# Plot the SMA for temperature

# colors for the line plot
colors = ["green", "red", "purple"]

# line plot - the yearly average air temperature in Barcelona
df_temperature.plot(color=colors, linewidth=2, figsize=(12, 6))

# modify ticks size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(
    labels=["Average air temperature", "10-years SMA", "20-years SMA"], fontsize=14
)

# title and labels
plt.title("The yearly average air temperature in Barcelona", fontsize=20)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Temperature [°C]", fontsize=16)


# Plot the SMA for rainfall

# colors for the line plot
colors = ["steelblue", "red", "purple"]

# line plot - the yearly accumulated rainfall in Barcelona
df_rainfall.plot(color=colors, linewidth=3, figsize=(12, 6))

# modify ticks size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=["Accumulated rainfall", "10-years SMA", "20-years SMA"], fontsize=14)

# title and labels
plt.title("The yearly accumulated rainfall in Barcelona", fontsize=20)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Rainfall [mm]", fontsize=16)


# The Cumulative Moving Average

# temperature

df_temperature["CMA"] = df_temperature.average_temperature.expanding().mean()
print(df_temperature)
# rainfall
df_rainfall["CMA"] = df_rainfall.accumulated_rainfall.expanding().mean()
print(df_rainfall)

# Plot showing CMA for temperature

# colors for the line plot
colors = ["green", "orange"]

# line plot - the yearly average air temperature in Barcelona
df_temperature[["average_temperature", "CMA"]].plot(
    color=colors, linewidth=3, figsize=(12, 6)
)

# modify ticks size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=["Average air temperature", "CMA"], fontsize=14)

# title and labels
plt.title("The yearly average air temperature in Barcelona", fontsize=20)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Temperature [°C]", fontsize=16)

# Plot showing CMA for rainfall

# colors for the line plot
colors = ["steelblue", "deeppink"]

# line plot - the yearly accumulated rainfall in Barcelona
df_rainfall[["accumulated_rainfall", "CMA"]].plot(
    color=colors, linewidth=2, figsize=(12, 6)
)

# modify ticks size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=["Accumulated rainfall", "CMA"], fontsize=14)

# title and labels
plt.title("The yearly accumulated rainfall in Barcelona", fontsize=20)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Rainfall [mm]", fontsize=16)


# Exponential Moving Average

# Temperature


# smoothing factor - 0.1
df_temperature["EMA_0.1"] = df_temperature.average_temperature.ewm(
    alpha=0.1, adjust=False
).mean()

# smoothing factor - 0.3
df_temperature["EMA_0.3"] = df_temperature.average_temperature.ewm(
    alpha=0.3, adjust=False
).mean()

print(df_temperature)

# rainfall
# smoothing factor - 0.1
df_rainfall["EMA_0.1"] = df_rainfall.accumulated_rainfall.ewm(
    alpha=0.1, adjust=False
).mean()

# smoothing factor - 0.3
df_rainfall["EMA_0.3"] = df_rainfall.accumulated_rainfall.ewm(
    alpha=0.3, adjust=False
).mean()
print(df_rainfall)

# Plotting EMA

# Temperature

# colors for the line plot
colors = ["green", "orchid", "orange"]

# line plot - the yearly average air temperature in Barcelona
df_temperature[["average_temperature", "EMA_0.1", "EMA_0.3"]].plot(
    color=colors, linewidth=2, figsize=(12, 6), alpha=0.8
)

# modify ticks size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(
    labels=["Average air temperature", "EMA - alpha=0.1", "EMA - alpha=0.3"],
    fontsize=14,
)

# title and labels
plt.title("The yearly average air temperature in Barcelona", fontsize=20)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Temperature [°C]", fontsize=16)


# Rainfall

# colors for the line plot
colors = ["steelblue", "coral", "green"]

# line plot - the yearly accumulated rainfall in Barcelona
df_rainfall[["accumulated_rainfall", "EMA_0.1", "EMA_0.3"]].plot(
    color=colors, linewidth=2, figsize=(12, 6), alpha=0.8
)

# modify ticks size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(
    labels=["Accumulated rainfall", "EMA - alpha=0.1", "EMA - alpha=0.3"], fontsize=14
)

# title and labels
plt.title("The yearly accumulated rainfall in Barcelona", fontsize=20)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Rainfall [mm]", fontsize=16)

# plt.show()

# As shown above, a small weighting factor α results in a high degree of
# smoothing, while a larger value provides a quicker response to recent changes.

# the weights of the simple and exponential moving averages (alpha=0.3, adjust=False) for 15 data points

# smoothing factor and number of data points
ALPHA = 0.3
N = 15

# weights - simple moving average
w_sma = np.repeat(1 / N, N)

# weights - exponential moving average alpha=0.3 adjust=False
w_ema = [(1 - ALPHA) ** i if i == N - 1 else ALPHA * (1 - ALPHA) ** i for i in range(N)]

# store the values in a data frame
pd.DataFrame({"w_sma": w_sma, "w_ema": w_ema}).plot(kind="bar", figsize=(10, 6))

# modify ticks size and labels
plt.xticks([])
plt.yticks(fontsize=14)
plt.legend(
    labels=["Simple moving average", "Exponential moving average (α=0.3)"], fontsize=14
)

# title and labels
plt.title("Moving Average Weights", fontsize=20)
plt.ylabel("Weights", fontsize=16)

plt.show()
