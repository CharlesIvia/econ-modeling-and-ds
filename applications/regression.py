import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qeds
from sklearn import linear_model, metrics, neural_network, pipeline, model_selection
from itertools import cycle
from sklearn.model_selection import cross_val_score

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
    ax.scatter(X[x], mod.predict(X), c=color, alpha=0.5, s=1)
    return ax


scatter_model(lr_model, X, ax, color=colors[1])
scatter_model(sqft_lr_model, X[["sqft_living"]], ax, color=colors[2])
ax.legend(["data", "full model", "sqft model"], markerscale=5)

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

# Lasso regression

lasso_model = linear_model.Lasso()
lasso_model.fit(X, y)

lasso_coefs = pd.Series(dict(zip(list(X), lasso_model.coef_)))
lr_coefs = pd.Series(dict(zip(list(X), lr_model.coef_)))

coefs = pd.DataFrame(dict(lasso=lasso_coefs, linreg=lr_coefs))

print(coefs)

# Compute lasso for may alphas (the lasso path)

alphas = np.exp(np.linspace(10, -2, 50))
alphas, coefs_lasso, _ = linear_model.lasso_path(
    X, y, alphas=alphas, fit_intercept=True, max_iter=1000
)

# plotting

fig, ax = plt.subplots(figsize=(15, 11))
colors = cycle(qeds.themes.COLOR_CYCLE)
log_alphas = -np.log10(alphas)

for coef_l, c, name in zip(coefs_lasso, colors, list(X)):
    ax.plot(log_alphas, coef_l, c=c)
    ax.set_xlabel("-Log(alpha)")
    ax.set_ylabel("lasso coefficients")
    ax.set_title("Lasso Path")
    ax.axis("tight")
    maxabs = np.max(np.abs(coef_l))
    i = [idx for idx in range(len(coef_l)) if abs(coef_l[idx]) >= (0.9 * maxabs)][0]
    xnote = log_alphas[i]
    ynote = coef_l[i]
    ax.annotate(name, (xnote, ynote), color=c)


# Overfitting and Regularization

# Split the data set into training and testing subsets.
# We will use the first 50 observations for training and the rest for testing.
# Fit the linear regression model and report MSE on training and testing datasets.
# Fit the lasso model and report the same statistics.


def fit_and_report_mses(mod, X_train, X_test, y_train, y_test):
    mod.fit(X_train, y_train)
    return dict(
        mse_train=metrics.mean_squared_error(y_train, mod.predict(X_train)),
        mse_test=metrics.mean_squared_error(y_test, mod.predict(X_test)),
    )


n_test = 50

X_train = X.iloc[:n_test, :]
print(X_train)

X_test = X.iloc[n_test:, :]
print(X_test)

y_train = y.iloc[:n_test]
print(y_train)

y_test = y.iloc[n_test:]
print(y_test)

result_lnr_mod = fit_and_report_mses(
    linear_model.LinearRegression(), X_train, X_test, y_train, y_test
)

print(result_lnr_mod)

result_lasso_mod = fit_and_report_mses(
    linear_model.Lasso(), X_train, X_test, y_train, y_test
)

print(result_lasso_mod)

# The MSE on the training dataset was smaller for the linear model without
# the regularization but the MSE on the test dataset was much higher.
# This strongly suggests that the linear regression model was OVERFITTING

alphas = np.exp(np.linspace(10, -5, 100))
mse = pd.DataFrame(
    [
        fit_and_report_mses(
            linear_model.Lasso(alpha=alpha, max_iter=50000),
            X_train,
            X_test,
            y_train,
            y_test,
        )
        for alpha in alphas
    ]
)
mse["log_alpha"] = -np.log10(alphas)
fig, ax = plt.subplots(figsize=(10, 6))
colors = qeds.themes.COLOR_CYCLE
mse.plot(x="log_alpha", y="mse_test", c=colors[0], ax=ax)
mse.plot(x="log_alpha", y="mse_train", c=colors[1], ax=ax)
ax.set_xlabel(r"$-\log(\alpha)$")
ax.set_ylabel("MSE")
ax.get_legend().remove()
ax.annotate("test", (mse.log_alpha[15], mse.mse_test[15]), color=colors[0])
ax.annotate("train", (mse.log_alpha[30], mse.mse_train[30]), color=colors[1])

# Crosss-validation of Regularization Parameter

# Partition the dataset randomly into k subsets/”folds”
# Compute  𝑀𝑆𝐸𝑗(𝛼)=  mean squared error in j-th subset when using the j-th subset as test data, and other k-1 as training data
# Minimize average (across folds) MSE  min𝛼 1/𝑘 ∑𝑘 𝑗=1 𝑀𝑆𝐸𝑗(𝛼)

mse["cv"] = [
    -np.mean(
        cross_val_score(
            linear_model.Lasso(alpha=alpha, max_iter=50000),
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_squared_error",
        )
    )
    for alpha in alphas
]

print(mse)

mse.plot(x="log_alpha", y="cv", c=colors[2], ax=ax)
ax.annotate("5 fold cross-validation", (mse.log_alpha[40], mse.cv[40]), color=colors[2])
ax.get_legend().remove()
ax.set_xlabel(r"$-\log(\alpha)$")
ax.set_ylabel("MSE")

# LassoCV exploits special structure of lasso problem to minimize CV more efficiently
lasso = linear_model.LassoCV(cv=5).fit(X_train, y_train)
print(
    -np.log10(lasso.alpha_)
)  # should roughly = minimizer on graph, not exactly equal due to random splitting

plt.show()
