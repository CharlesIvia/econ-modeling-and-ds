import matplotlib.colors as mplc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from pandas_datareader import DataReader
import qeds

qeds.themes.mpl_style()

# Step 0: Research

# Identify a concise message

##WANT: Determine what happened to rural/urban wage gap for non-college educated workers since the 1950's

# Visualization  draft


# Read in data
df = pd.read_csv("https://datascience.quantecon.org/assets/data/density_wage_data.csv")
df["year"] = df.year.astype(int)  # Convert year to int


def single_scatter_plot(df, year, educ, ax, color):
    """
    This function creates a single year's and education level's
    log density to log wage plot
    """
    # Filter data to keep only the data of interest
    _df = df.query("(year == @year) & (group == @educ)")
    _df.plot(kind="scatter", x="density_log", y="wages_logs", ax=ax, color=color)

    return ax


# Create initial plot
# fig, ax = plt.subplots(1, 4, figsize=(16, 6), sharey=True)

# for (i, year) in enumerate(df.year.unique()):
#     single_scatter_plot(df, year, "college", ax[i], "b")
#     single_scatter_plot(df, year, "noncollege", ax[i], "r")
#     ax[i].set_title(str(year), fontsize=10)


# Fine tuning the plot

fig, ax = plt.subplots(1, 4, figsize=(16, 6))
colors = {"college": "#1385ff", "noncollege": "#ff6d13"}

for (i, year) in enumerate(df.year.unique()):
    single_scatter_plot(df, year, "college", ax[i], colors["college"])
    single_scatter_plot(df, year, "noncollege", ax[i], colors["noncollege"])
    ax[i].set_title(str(year), fontsize=10)

bgcolor = (250 / 255, 250 / 255, 250 / 255)
fig.set_facecolor(bgcolor)
for (i, _ax) in enumerate(ax):
    # Label with words
    if i == 0:
        _ax.set_xlabel("Population Density", fontsize=10)
    else:
        _ax.set_xlabel("")

    # Turn off right and top axis lines
    _ax.spines["right"].set_visible(False)
    _ax.spines["top"].set_visible(False)

    # Don't use such a white background color
    _ax.set_facecolor(bgcolor)

    # Change bounds
    _ax.set_ylim((np.log(4), np.log(30)))
    _ax.set_xlim((0, 10))

    # Change ticks
    xticks = [10, 100, 1000, 10000]
    _ax.set_xticks([np.log(xi) for xi in xticks])
    _ax.set_xticklabels([str(xi) for xi in xticks])

    yticks = list(range(5, 32, 5))
    _ax.set_yticks([np.log(yi) for yi in yticks])
    if i == 0:
        _ax.set_yticklabels([str(yi) for yi in yticks])
        _ax.set_ylabel("Average Wage", fontsize=10)
    else:
        _ax.set_yticklabels([])
        _ax.set_ylabel("")

ax[0].annotate(
    "College Educated Workers", (np.log(75), np.log(14.0)), color=colors["college"]
)
ax[0].annotate(
    "Non-College Educated Workers",
    (np.log(10), np.log(5.25)),
    color=colors["noncollege"],
)
ax[0].set_zorder(1)

fig.tight_layout()
plt.show()
