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
