# WANT: Explore SMA, CMA, EMA usinf rainfall and temperature data from Open Data Barcelona

import pandas as pd
import matplotlib.pyplot as plt

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

plt.show()
