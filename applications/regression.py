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

plt.show()
