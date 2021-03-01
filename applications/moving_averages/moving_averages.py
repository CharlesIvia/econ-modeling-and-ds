# WANT: Explore SMA, CMA, EMA usinf rainfall and temperature data from Open Data Barcelona

import pandas as pd

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

df_rainfall["average_rainfall"] = df_rainfall.mean(axis=1)
print(df_rainfall)

# drop monthly values

df_temperature = df_temperature[["average_temperature"]]
print(df_temperature)

df_rainfall = df_rainfall[["average_rainfall"]]
print(df_rainfall)
