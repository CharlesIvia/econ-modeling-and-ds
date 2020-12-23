import numpy as np
import pandas as pd
import qeds
import matplotlib.pyplot as plt

qeds.themes.mpl_style()

# Split-Apply-Combine

# Split: split the data into groups based on values in one or more columns.
# Apply: apply a function or routine to each group separately.
# Combine: combine the output of the apply step into a DataFrame, using the group identifiers as the index.

C = np.arange(1, 7, dtype=float)
print(C)
C[[3, 5]] = np.nan
print(C)

df = pd.DataFrame(
    {
        "A": [1, 1, 1, 2, 2, 2],
        "B": [1, 1, 2, 2, 1, 1],
        "C": C,
    }
)

print(df)

# Step 1 - call groupby method to set up split

gbA = df.groupby("A")
print(gbA)

print(gbA.get_group(2))

# If we pass a list of strings to groupby, it will group based on unique
# combinations of values from all columns in the list.

gbAB = df.groupby(["A", "B"])
print(gbAB.get_group((1, 1)))

print(gbAB.count())

# CASE STUDY: Airline Delays

air_dec = qeds.load("airline_performance_dec16")
print(air_dec)

weekly_delays = (
    air_dec.groupby([pd.Grouper(key="Date", freq="W"), "Carrier"])[
        "ArrDelay"
    ]  # extract one column
    .mean()  # take average
    .unstack(level="Carrier")  # Flip carrier up as column names
)

print(weekly_delays)

delay_cols = [
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

pre_christmas = air_dec.loc[
    (air_dec["Date"] >= "2016-12-12") & (air_dec["Date"] <= "2016-12-18")
]

# custom agg function


def positive(df):
    return (df > 0).sum()


delay_totals = pre_christmas.groupby("Carrier")[delay_cols].agg(
    ["sum", "mean", positive]
)
print(delay_totals)

reshaped_delays = (
    delay_totals.stack()  # move aggregation method into index (with Carrier)
    .T.swaplevel(  # put delay type in index and Carrier+agg in column
        axis=1
    )  # make agg method outer level of column label
    .sort_index(axis=1)  # sort column labels so it prints nicely
)
print(reshaped_delays)

for agg in ["mean", "sum", "positive"]:
    axs = reshaped_delays[agg].plot(
        kind="bar",
        subplots=True,
        layout=(4, 3),
        figsize=(10, 8),
        legend=False,
        sharex=True,
        sharey=True,
    )
    fig = axs[0, 0].get_figure()
    fig.suptitle(agg)
    fig.tight_layout()

plt.show()
