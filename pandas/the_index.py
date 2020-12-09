import pandas as pd
import numpy as np

# Load data

url = "https://datascience.quantecon.org/assets/data/wdi_data.csv"
df = pd.read_csv(url)
print(df.info())

print(df.head())

df_small = df.head(5)
print(df_small)

df_tiny = df.iloc[[0, 3, 2, 4], :]
print(df_tiny)

im_ex = df_small[["Imports", "Exports"]]
im_ex_copy = im_ex.copy()
print(im_ex_copy)

print(im_ex + im_ex_copy)

im_ex_tiny = df_tiny + im_ex
print(im_ex_tiny)
