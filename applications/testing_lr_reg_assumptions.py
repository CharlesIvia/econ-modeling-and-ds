# Testing linear regression assumptions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# data - features, predictors, label - target/label/response variable

boston = datasets.load_boston()

# Fake linear data for comparing with actual

linear_X, linear_y = datasets.make_regression(
    n_samples=boston.data.shape[0],
    n_features=boston.data.shape[1],
    noise=75,
    random_state=46,
)

# Setting feature names to x1, x2, x3, etc. if they are not defined
linear_feature_names = ["X" + str(feature + 1) for feature in range(linear_X.shape[1])]


df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["HousePrice"] = boston.target

print(df.head())

# Initial setup

# Fitting the boston model

boston_model = LinearRegression()
boston_model.fit(boston.data, boston.target)

# Returning the R^2 for the model
boston_r2 = boston_model.score(boston.data, boston.target)
print("R^2: {0}".format(boston_r2))

# Fitting the other model

# Fitting the model
linear_model = LinearRegression()
linear_model.fit(linear_X, linear_y)

# Returning the R^2 for the model
linear_r2 = linear_model.score(linear_X, linear_y)
print("R^2: {0}".format(linear_r2))


def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({"Actual": label, "Predicted": predictions})
    df_results["Residuals"] = abs(df_results["Actual"]) - abs(df_results["Predicted"])

    return df_results
