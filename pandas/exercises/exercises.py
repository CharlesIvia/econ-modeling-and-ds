#Import required libraries
import pandas as pd
from datetime import datetime

hotels = pd.read_csv("hotel_booking_data.csv")

print(hotels.head())

# How many rows are there?
rows = hotels.shape[0]
print(rows)

# Is there any missing data? If so, which column has the most missing data?
print(hotels.isna().sum())

# Drop the "company" column from the dataset.
print(hotels.drop("company", axis=1).columns)

# What are the top 5 most common country codes in the dataset?
print(hotels["country"].value_counts())

# What is the name of the person who paid the highest ADR (average daily rate)? How much was their ADR?
print(hotels.iloc[hotels["adr"].sort_values(ascending=False).index[0]])

# The adr is the average daily rate for a person's stay at the hotel. What is the mean adr across all the hotel stays in the dataset?

print(hotels["adr"].mean())

# What is the average (mean) number of nights for a stay across the entire data set? Feel free to round this to 2 decimal points.

print(
    round(
        (hotels["stays_in_weekend_nights"] + hotels[
         "stays_in_week_nights"]).mean(), 2
    )
)

# What is the average total cost for a stay in the dataset?

hotels["total_cost"] = (
    hotels["stays_in_weekend_nights"] + hotels["stays_in_week_nights"]
) * hotels["adr"]

print(round(hotels["total_cost"].mean(), 2))

more_than_5 = hotels[hotels["total_of_special_requests"] == 5]
print(more_than_5[["name", "email"]])

# What percentage of hotel stays were classified as "repeat guests"?
print(hotels["is_repeated_guest"].mean() * 100)
print(
    (
        hotels[hotels["is_repeated_guest"] >= 1].sum()["is_repeated_guest"]
        / hotels["is_repeated_guest"].count()
    )
    * 100
)

# What are the top 5 most common last name in the dataset?

print((hotels["name"].str.split(" ").str[-1]).value_counts()[:5])

# What are the names of the people who had booked the most number children and babies for their stay?

hotels["total_kids"] = hotels["babies"] + hotels["children"]
most_kids = hotels.sort_values(by="total_kids", ascending=False)[:3]
print(most_kids[["name", "adults", "total_kids", "babies", "children"]])

# What are the top 3 most common area code in the phone numbers?

top_codes = hotels["phone-number"].str[:3].value_counts()[:3]
print(f"Code - Total Count \n{top_codes}")

# How many arrivals took place between the 1st and the 15th of the month

print(hotels[hotels["arrival_date_day_of_month"].isin(range(1, 16))]
      .name.count())

# Create a table for counts for each day of the week that people arrived.

print(hotels.columns)
hotels["arrival_date"] = (
    hotels["arrival_date_year"].astype(str)
    + "-"
    + hotels["arrival_date_month"].astype(str)
    + "-"
    + hotels["arrival_date_day_of_month"].astype(str)
)
hotels["arrival_date"] = pd.to_datetime(hotels["arrival_date"])

hotels["weekday"] = hotels["arrival_date"].dt.day_name()
print(hotels["arrival_date"])
print(hotels["weekday"])

print(hotels["weekday"].value_counts())
