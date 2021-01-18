import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qeds
from sklearn import linear_model, metrics, neural_network, pipeline, model_selection

qeds.themes.mpl_style()
plotly_template = qeds.themes.plotly_template()
colors = qeds.themes.COLOR_CYCLE

url = "https://datascience.quantecon.org/assets/data/kc_house_data.csv"
df = pd.read_csv(url)
print(df.info())

X = df.drop(["price", "date", "id"], axis=1).copy()

# conver everything to a float

for col in list(X):
    X[col] = X[col].astype(float)
print(X.head())

y = np.log(df["price"])
df["log_price"] = y
print(y.head())


def var_scatter(df, ax=None, var="sqft_living"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    df.plot.scatter(x=var, y="log_price", alpha=0.35, s=1.5, ax=ax)

    return ax


var_scatter(df)

# Linear regression
sns.lmplot(
    data=df,
    x="sqft_living",
    y="log_price",
    height=5,
    scatter_kws=dict(s=1.5, alpha=0.35),
)

# Using sklearn to replicate above figures

# Construct model instance

sqft_lr_model = linear_model.LinearRegression()

# fit the model

sqft_lr_model.fit(X[["sqft_living"]], y)

# Print the coefficients

beta_0 = sqft_lr_model.intercept_
beta_1 = sqft_lr_model.coef_[0]

print(f"Fit model: log(price) = {beta_0:.4f} + {beta_1:.4f} sqft_living")

# Construct the plot

ax = var_scatter(df)

# ponts for the line

x = np.array([0, df["sqft_living"].max()])
ax.plot(x, beta_0 + beta_1 * x)

# Predicting the price if a 5000 sqft home

logp_5000 = sqft_lr_model.predict([[5000]])[0]

print(
    f"The model predicts a 5,000sq. foot home would cost {np.exp(logp_5000):.2f} dollars"
)

# fit the linear regression model using all columns in X

lr_model = linear_model.LinearRegression()
lr_model.fit(X, y)

# Visualizing impact of extra variables on the lr model

ax = var_scatter(df)


def scatter_model(mod, X, ax=None, color=colors[1], x="sqft_living"):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(X[x], mod.predict(X), c=color, alpha=0.25, s=1)
    return ax


scatter_model(lr_model, X, ax, color=colors[1])
scatter_model(sqft_lr_model, X[["sqft_living"]], ax, color=colors[2])
ax.legend(["data", "full model", "sqft model"])

# Nonlinear Relationships in Linear regression

X2 = X[["sqft_living"]].copy()

X2["pct_sqft_above"] = X["sqft_above"] / X["sqft_living"]
print(X2)


sqft_above_lr_model = linear_model.LinearRegression()
sqft_above_lr_model.fit(X2, y)

new_mse = metrics.mean_squared_error(y, sqft_above_lr_model.predict(X2))
old_mse = metrics.mean_squared_error(y, sqft_lr_model.predict(X2[["sqft_living"]]))
print(
    f"The mse changed from {old_mse:.4f} to {new_mse:.4f} by including our new feature"
)
# plt.show()
