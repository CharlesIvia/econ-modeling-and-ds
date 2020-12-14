from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
import pyarrow.feather

# FILE FORMATS

# Csv, Xls, Parquet, Feather, SQL

# WRITING DAFAFRAMES

np.random.seed(42)
df1 = pd.DataFrame(
    np.random.randint(0, 100, size=(10, 4)), columns=["a", "b", "c", "d"]
)

print(df1)

wanted_mb = 10
nrow = 100000
ncol = int(((wanted_mb * 1024 ** 2) / 8) / nrow)
df2 = pd.DataFrame(
    np.random.rand(nrow, ncol), columns=["x{}".format(i) for i in range(ncol)]
)

print("df2.shape = ", df2.shape)
print("df2 is approximately {} MB".format(df2.memory_usage().sum() / (1024 ** 2)))

# to_csv
df1.to_csv("df1_.csv")

# to_excel

df1.to_excel("df1.xlsx", "df1")

# more than one dataframe to a workbook using a context manager

with pd.ExcelWriter("df1.xlsx") as writer:
    df1.to_excel(writer, "df1")
    (df1 + 10).to_excel(writer, "df1 plus 10")

# pyarrow.feather.write_feather
pyarrow.feather.write_feather(df1, "df1.feather")

# READING FILES INTO DATAFRAMES

# read csv
df1_csv = pd.read_csv("./pandas/df1.csv", index_col=0)
print(df1_csv.head())

# read xls
df1_xlsx = pd.read_excel("./pandas/df1.xlsx", "df1", index_col=0)
print(df1_xlsx.head())

# feather
df1_feather = pyarrow.feather.read_feather("./pandas/df1.feather")
print(df1_feather.head())

# read internet-stored file
url = "https://raw.githubusercontent.com/fivethirtyeight/nfl-elo-game/"
url = url + "3488b7d0b46c5f6583679bc40fb3a42d729abd39/data/nfl_games.csv"
df1_web = pd.read_csv(url, index_col=0)
print(df1_web.head())
