import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qeds
from sklearn import linear_model, metrics, neural_network, pipeline, model_selection
from sklearn.model_selection import cross_val_score

qeds.themes.mpl_style()

# Introduction to Recidivism

# Recidivism is the tendency for an individual who has previously committed a
# crime to commit another crime in the future.

data_url = "https://raw.githubusercontent.com/propublica/compas-analysis"
data_url += "/master/compas-scores-two-years.csv"

df = pd.read_csv(data_url)
print(df.head())
print(df.describe())

# Variables used

# first: An individual’s first name
# last: An individual’s last name
# sex: An individual’s sex
# age: An individual’s age
# race: An individual’s race. It takes values of Caucasian, Hispanic, African-American,
# ...Native American, Asian, or Other
# priors_count: Number of previous arrests
# decile_score: The COMPAS risk score
# two_year_recid: Whether the individual had been jailed for a new crime in next two years


# Descriptive statistics

# Keep only data on race with at least 500 observations
# Remember this can still perpetuate inequality by exclusion
race_count = df.groupby(["race"])["name"].count()
print(race_count)

at_least_500 = list(race_count[race_count > 500].index)
print(at_least_500)

df = df.loc[df["race"].isin(at_least_500), :]
print(df.head())

# Age, Sex and Race breakdown


def create_groupcount_barplot(df, group_col, figsize, **kwargs):
    "call df.groupby(group_col), then count number of records and plot"
    counts = df.groupby(group_col)["name"].count().sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind="bar", **kwargs)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    return fig, ax


# By age

age_cs = ["Less than 25", "25 - 45", "Greater than 45"]
df["age_cat"] = pd.Categorical(df["age_cat"], categories=age_cs, ordered=True)
fig, ax = create_groupcount_barplot(df, "age_cat", (14, 8), color="DarkBlue", rot=0)

# By sex
sex_cs = ["Female", "Male"]
df["sex"] = pd.Categorical(df["sex"], categories=sex_cs, ordered=True)
create_groupcount_barplot(df, "sex", (6, 8), color="DarkBlue", rot=0)

# By race

race_cs = ["African-American", "Caucasian", "Hispanic"]
df["race"] = pd.Categorical(df["race"], categories=race_cs, ordered=True)
create_groupcount_barplot(df, "race", (12, 8), color="DarkBlue", rot=0)

# From these plots we learn that the population is mostly between 25-45, male, and
# ...mostly African-American or Caucasian
plt.show()
