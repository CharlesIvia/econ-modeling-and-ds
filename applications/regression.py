import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qeds
from sklearn import (
    linear_model, metrics, neural_network, pipeline, model_selection
)

qeds.themes.mpl_style();
plotly_template = qeds.themes.plotly_template()
colors = qeds.themes.COLOR_CYCLE

url = "https://datascience.quantecon.org/assets/data/kc_house_data.csv"
df = pd.read_csv(url)
print(df.info())

X = df.drop(["price", "date", "id"], axis=1).copy()

#conver everything to a float

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

plt.show()