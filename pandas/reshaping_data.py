import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TIDY DATA

# Each variable forms a column.
# Each observation forms a row e.g. a country, a yeare.t.c
# Each type of observational unit forms a table.

# RESHAPING DATA

# shape is no of r and c's in a df

url = "https://datascience.quantecon.org/assets/data/bball.csv"
bball = pd.read_csv(url)

print(bball.info())
print(bball)

# Long v Wide DataFrames

bball_long = bball.melt(id_vars=["Year", "Player", "Team", "TeamName"])
print(bball_long)

bball_wide = bball_long.pivot_table(
    index="Year", columns=["Player", "variable", "Team"], values="value"
)
print(bball_wide)

# set_index, reset_index and Transpose

# set_index: Move one or more columns into the index.
# reset_index: Move one or more index levels out of the index and make them either columns or drop from DataFrame.
# T: Swap row and column labels.

bball2 = bball.set_index(["Player", "Year"])
print(bball2)

bball3 = bball2.T
print(bball3.head())

# stack and unstack

# stack is used to move certain levels of the column labels into the index (i.e. moving from wide to long)

print(bball_wide.stack())

player_stats = bball_wide.stack().mean()

print(player_stats)

# move the Player level down into the index so we are left with column levels for Team and variable
team_stats = bball_wide.stack(level="Player").mean()
print(team_stats)

# Without any arguments, the stack arguments move the level of column labels closest to the data
# (also called inner-most or bottom level of labels) to become
# the index level closest to the data (also called the inner-most or right-most level of the index).
# In our example, this moved Team down from columns to the index.


# When we do pass a level, that level of column labels is moved down to the right-most
# level of the index and all other column labels stay in their relative position.


print(bball_wide.stack(level=["Player", "Team"]))

# unstack

##Want: To see a bar chart for each players starts
# we will need to have the player’s name on the index and the variables as columns to do this.
print(player_stats)

# We now need to rotate the variable level of the index up to be column layers.

print(player_stats.unstack())

player_stats.unstack().plot.bar()

# Compare all players for each statistic

print(player_stats.unstack(level="Player"))

player_stats.unstack(level="Player").plot.bar()

plt.tight_layout()
# plt.show()

# It is the inverse of stack; stack will move labels down from columns to index,
# while unstack moves them up from index to columns.
# By default, unstack will move the level of the index closest to the data
# and place it in the column labels closest to the data.

# Pro tip: We remember stack vs unstack with a mnemonic: Unstack moves index levels Up

# MELT

# used to move from wide to long

# Warning: When you use melt, any index that you currently have will be deleted.

print(bball.melt(id_vars=["Year", "Player", "Team", "TeamName"]))

# the columns we specified as id_vars remained columns, but all other columns were put into two new columns:
# variable: This has dtype string and contains the former column names. as values
# value: This has the former values.


# PIVOT AND PIVOT_ABLE

# pivot

print(bball)

print(bball.head(6).pivot(index="Year", columns="Player", values="Pts"))

# can replicate above pivot as below

print(bball.head(6).set_index(["Year", "Player"])["Pts"].unstack(level="Player"))

# Note: in order for pivot to work, the index/column pairs must be unique!

# pivot_table

print(bball.pivot_table(index=["Year", "Team"], columns="Player", values="Pts"))

print(bball.pivot_table(index="Year", columns=["Player", "Team"], values="Pts"))

bball_pivoted = bball.pivot_table(index="Year", columns="Player", values="Pts")
print(bball_pivoted)

# pivot_table handles duplicate index/column pairs using an aggregation.
# default = mean

# We can choose how pandas aggregates all of the values.

# Using max
print(bball.pivot_table(index="Year", columns="Player", values="Pts", aggfunc=max))

# Count no of values

print(bball.pivot_table(index="Year", columns="Player", values="Pts", aggfunc=len))

# multiple aggregation functions

print(
    bball.pivot_table(index="Year", columns="Player", values="Pts", aggfunc=[max, len])
)

# VISUALIZING RESHAPING

# made up
# columns A and B are "identifiers" while C, D, and E are variables.
df = pd.DataFrame(
    {
        "A": [0, 0, 1, 1],
        "B": "x y x z".split(),
        "C": [1, 2, 1, 4],
        "D": [
            10,
            20,
            30,
            20,
        ],
        "E": [
            2,
            1,
            5,
            4,
        ],
    }
)

print(df.info())
print(df)

df2 = df.set_index(["A", "B"])
print(df2)

df3 = df2.T
print(df3.head())
