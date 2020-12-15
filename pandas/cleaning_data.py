import pandas as pd
import numpy as np
import qeds

df = pd.DataFrame(
    {
        "numbers": ["#23", "#24", "#18", "#14", "#12", "#10", "#35"],
        "nums": ["23", "24", "18", "14", np.nan, "XYZ", "35"],
        "colors": ["green", "red", "yellow", "orange", "purple", "blue", "pink"],
        "other_column": [0, 1, 0, 2, 1, 0, 2],
    }
)

print(df)

# this generates an error - coul not convert ... to numeric

# print(df["numbers"].mean())

# STRING METHODS

# int(replace("#", ""))

# Slow method - iterating over all rows

for row in df.iterrows():
    index_value, column_values = row

    clean_number = int(column_values["numbers"].replace("#", " "))
    df.at[index_value, "numbers_loop"] = clean_number

print(df["numbers_loop"].mean())

# Faster method - apply a sting method to an entire column of data
# s.str.method_name

df["numbers_str"] = df["numbers"].str.replace("#", " ")
print(df)

# We can use .str to access almost any string method that works on normal strings

print(df["colors"].str.contains("p"))
print(df["colors"].str.capitalize())

# TYPE CONVERSIONS

print(df["numbers_str"].dtype)

# convert to numbers using pd.to_numeric

df["numbers_numeric"] = pd.to_numeric(df["numbers_str"])
print(df["numbers_numeric"].dtype)

# using astype() we can convert to any supported stype

df["numbers_numeric"].astype(str)
df["numbers_numeric"].astype(float)
