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
