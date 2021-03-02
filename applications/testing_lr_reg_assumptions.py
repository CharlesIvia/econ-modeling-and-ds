# Testing linear regression assumptions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


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
print(f"R^2: {boston_r2}")

# Fitting the other model

# Fitting the model
linear_model = LinearRegression()
linear_model.fit(linear_X, linear_y)

# Returning the R^2 for the model
linear_r2 = linear_model.score(linear_X, linear_y)
print(f"R^2: {linear_r2}")


def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({"Actual": label, "Predicted": predictions})
    df_results["Residuals"] = abs(df_results["Actual"]) - abs(df_results["Predicted"])

    return df_results


# Assumptions

# First Assumption - Linearlity
# This assumes that there is a linear relationship between the predictors
# (e.g. independent variables or features)
# and the response variable (e.g. dependent variable or label).
# the predictors are additive.

# How? use a scatter plot to see if residuals lie around a diagonal line on the scatter plot


def linear_assumption(model, features, label):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """
    print("Assumption 1: Linear Relationship between the Target and the Feature", "\n")

    print(
        "Checking with a scatter plot of actual vs. predicted.",
        "Predictions should follow the diagonal line.",
    )

    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the actual vs predicted values
    sns.lmplot(
        x="Actual", y="Predicted", data=df_results, fit_reg=False, height=6, aspect=2
    )

    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(
        line_coords, line_coords, color="darkorange", linestyle="--"  # X and y points
    )
    plt.title("Actual vs. Predicted")
    plt.tight_layout()
    plt.show()


# Linearlity in the linear dataset

linear_assumption(linear_model, linear_X, linear_y)

# Linearlity in the boston dataset
linear_assumption(boston_model, boston.data, boston.target)


# Second assumption- normality of the error terms

# this assumes that the error terms of the model are normally distributed

# Cause of non-normality - if ind or dependent variables are significantly non-normal

# Test - a histogram and the p-value from the Anderson-Darling test for normality.

# Fix - nonlinear variable transformations, exluding some varibales (lon-tailed variables) or removing outliers


def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.

    This assumption being violated primarily causes issues with the confidence intervals
    """
    print("Assumption 2: The error terms are normally distributed", "\n")

    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)

    print("Using the Anderson-Darling test for normal distribution")

    # Performing the test on the residuals
    p_value = normal_ad(df_results["Residuals"])[1]
    print("p-value from the test - below 0.05 generally means non-normal:", p_value)

    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print("Residuals are not normally distributed")
    else:
        print("Residuals are normally distributed")

    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title("Distribution of Residuals")
    sns.distplot(df_results["Residuals"])
    plt.show()

    print()
    if p_value > p_value_thresh:
        print("Assumption satisfied")
    else:
        print("Assumption not satisfied")
        print()
        print("Confidence intervals will likely be affected")
        print("Try performing nonlinear transformations on variables")


# Normality test for the linear dataset

normal_errors_assumption(linear_model, linear_X, linear_y)
normal_errors_assumption(boston_model, boston.data, boston.target)

# Third Assumption -  No Multicollinearity among Predictors

# This assumes that the predictors used in the regression are not correlated with each other

# This won’t render our model unusable if violated, but it will cause issues with the interpretability of the model


# Detection: use a heatmap of the correlation as a visual aid and examine the variance inflation factor (VIF)


def multicollinearity_assumption(model, features, label, feature_names=None):
    """
    Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                       correlation among the predictors, then either remove prepdictors with high
                       Variance Inflation Factor (VIF) values or perform dimensionality reduction

                       This assumption being violated causes issues with interpretability of the
                       coefficients and the standard errors of the coefficients.
    """
    print("Assumption 3: Little to no multicollinearity among predictors")

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(features, columns=feature_names).corr(), annot=True)
    plt.title("Correlation of Variables")
    plt.show()

    print("Variance Inflation Factors (VIF)")
    print("> 10: An indication that multicollinearity may be present")
    print("> 100: Certain multicollinearity among the variables")
    print("-------------------------------------")

    # Gathering the VIF for each variable
    VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
    for idx, vif in enumerate(VIF):
        print("{0}: {1}".format(feature_names[idx], vif))

    # Gathering and printing total cases of possible or definite multicollinearity
    possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
    definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
    print()
    print(f"{possible_multicollinearity} cases of possible multicollinearity.")
    print(f"{definite_multicollinearity} cases of definite multicollinearity.")
    print()

    if definite_multicollinearity == 0:
        if possible_multicollinearity == 0:
            print("Assumption satisfied")
        else:
            print("Assumption possibly satisfied")
            print()
            print("Coefficient interpretability may be problematic")
            print(
                "Consider removing variables with a high Variance Inflation Factor (VIF)"
            )

    else:
        print("Assumption not satisfied")
        print()
        print("Coefficient interpretability will be problematic")
        print("Consider removing variables with a high Variance Inflation Factor (VIF)")


# Multicollinearlity in the linear dataset

multicollinearity_assumption(linear_model, linear_X, linear_y, linear_feature_names)

# Multicollinearlity in the boston dataset

multicollinearity_assumption(
    boston_model, boston.data, boston.target, boston.feature_names
)


# Fourth Assumption: No Autocorrelation of the error terms

# Autocorrelation being present typically indicates that we are missing some information that should be captured by the model

# Why it can happen?

# In a time series scenario, there could be information about the past that
# we aren’t capturing. In a non-time series scenario, our model could be systematically
# biased by either under or over predicting in certain conditions.
# Lastly, this could be a result of a violation of the linearity assumption.

# This will impact our model estimates.

# Detection: perform a Durbin-Watson test to determine if either positive or negative correlation is present. Alternatively, you could create plots of residual autocorrelations.

# Fix: Adding lag variables can fix this problem.
# Alternatively, interaction terms, additional variables, or additional transformations may fix thi


def autocorrelation_assumption(model, features, label):
    """
    Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                     autocorrelation, then there is a pattern that is not explained due to
                     the current value being dependent on the previous value.
                     This may be resolved by adding a lag variable of either the dependent
                     variable or some of the predictors.
    """
    print("Assumption 4: No Autocorrelation", "\n")

    # Calculating residuals for the Durbin Watson-tests
    df_results = calculate_residuals(model, features, label)

    print("\nPerforming Durbin-Watson Test")
    print(
        "Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data"
    )
    print("0 to 2< is positive autocorrelation")
    print(">2 to 4 is negative autocorrelation")
    print("-------------------------------------")
    durbinWatson = durbin_watson(df_results["Residuals"])
    print("Durbin-Watson:", durbinWatson)
    if durbinWatson < 1.5:
        print("Signs of positive autocorrelation", "\n")
        print("Assumption not satisfied")
    elif durbinWatson > 2.5:
        print("Signs of negative autocorrelation", "\n")
        print("Assumption not satisfied")
    else:
        print("Little to no autocorrelation", "\n")
        print("Assumption satisfied")


# Testing for autocorrelation in the linear dataset

autocorrelation_assumption(linear_model, linear_X, linear_y)

# Testing for autocorrelation in the boston dataset

autocorrelation_assumption(boston_model, boston.data, boston.target)
