# Many data sources contain both numerical data and text
# Text can be used to create features for any prediction model but only
# ...after encoding text into some numerical reprentation


# AVALANChES

# Avalanches are a hazard in mountains and can be predicted based on
# ...snow conditions, weather and terrain

# WANT: Predict fatal accidents from the text of avalanche forecasts

# Since fatal accidents are rare, this prediction task will be quite difficult


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qeds

# activate plot theme
qeds.themes.mpl_style()
