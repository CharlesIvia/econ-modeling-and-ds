import os
import pandas as pd
import matplotlib.pyplot as plt
import quandl
import qeds

# API key
quandl.ApiConfig.api_key = os.environ.get(
    "QUANDL_AUTH", "Dn6BtVoBhzuKTuyo6hbp")
start_date = "2014-05-01"

qeds.themes.mpl_style()

print(quandl.ApiConfig.api_key)

# Parsing Strings as Dates

christmas_str = "2020-12-25"
christmas = pd.to_datetime(christmas_str)

print(type(christmas))


for date in [
    "December 25, 2017",
    "Dec. 25, 2017",
    "Monday, Dec. 25, 2017",
    "25 Dec. 2017",
    "25th Dec. 2017",
]:
    print("pandas interprets {} as {}".format(date, pd.to_datetime(date)))


# Give pandas as hint

christmas_amzn = "2017-12-25T00:00:00+ 00 :00"

# print(pd.to_datetime(christmas_amzn))

amzn_strftime = "%Y-%m-%dT%H:%M:%S+ 00 :00"

print(pd.to_datetime(christmas_amzn, format=amzn_strftime))

# Notice

print(amzn_strftime)
print(christmas_amzn)

# multiple dates

print(pd.to_datetime(["2017-12-25", "2017-12-31"]))


#Date formatting

print(christmas.strftime("We love %A %B %d (also written  %c)"))