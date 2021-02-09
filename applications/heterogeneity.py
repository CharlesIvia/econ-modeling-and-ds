import pandas as pd
import numpy as np
import patsy
from sklearn import linear_model, ensemble, base, neural_network
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import matplotlib.pyplot as plt
import seaborn as sns


# Read data

url = "https://datascience.quantecon.org/assets/data/Triyana_2016_price_women_clean.csv.gz"
df = pd.read_csv(url)
print(df.describe())

# data prep

formula = """
bw ~ pkh_kec_ever +
  C(edu)*C(agecat) + log_xp_percap + hh_land + hh_home + C(dist) +
  hh_phone + hh_rf_tile + hh_rf_shingle + hh_rf_fiber +
  hh_wall_plaster + hh_wall_brick + hh_wall_wood + hh_wall_fiber +
  hh_fl_tile + hh_fl_plaster + hh_fl_wood + hh_fl_dirt +
  hh_water_pam + hh_water_mechwell + hh_water_well + hh_water_spring + hh_water_river +
  hh_waterhome +
  hh_toilet_own + hh_toilet_pub + hh_toilet_none +
  hh_waste_tank + hh_waste_hole + hh_waste_river + hh_waste_field +
  hh_kitchen +
  hh_cook_wood + hh_cook_kerosene + hh_cook_gas +
  tv + fridge + motorbike + car + goat + cow + horse
"""
bw, X = patsy.dmatrices(formula, df, return_type="dataframe")
# some categories are empty after dropping rows will Null, drop now
X = X.loc[:, X.sum() > 0]
bw = bw.iloc[:, 0]
treatment_variable = "pkh_kec_ever"
treatment = X["pkh_kec_ever"]
Xl = X.drop(["Intercept", "pkh_kec_ever", "C(dist)[T.313175]"], axis=1)
# scale = bw.std()
# center = bw.mean()
loc_id = df.loc[X.index, "Location_ID"].astype("category")

import re

# remove [ ] from names for compatibility with xgboost
Xl = Xl.rename(columns=lambda x: re.sub("\[|\]", "_", x))


# Estimate average treatment effects
from statsmodels.iolib.summary2 import summary_col

tmp = pd.DataFrame(
    dict(
        birthweight=bw,
        treatment=treatment,
        assisted_delivery=df.loc[X.index, "good_assisted_delivery"],
    )
)
usage = smf.ols("assisted_delivery ~ treatment", data=tmp).fit(
    cov_type="cluster", cov_kwds={"groups": loc_id}
)
health = smf.ols("bw ~ treatment", data=tmp).fit(
    cov_type="cluster", cov_kwds={"groups": loc_id}
)
print(summary_col([usage, health]))


# for clustering standard errors
def get_treatment_se(fit, cluster_id, rows=None):
    if cluster_id is not None:
        if rows is None:
            rows = [True] * len(cluster_id)
        vcov = sm.stats.sandwich_covariance.cov_cluster(fit, cluster_id.loc[rows])
        return np.sqrt(np.diag(vcov))

    return fit.HC0_se
