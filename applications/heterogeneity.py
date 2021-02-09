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


# Creating generic ml model


def generic_ml_model(x, y, treatment, model, n_split=10, n_group=5, cluster_id=None):
    nobs = x.shape[0]

    blp = np.zeros((n_split, 2))
    blp_se = blp.copy()
    gate = np.zeros((n_split, n_group))
    gate_se = gate.copy()

    baseline = np.zeros((nobs, n_split))
    cate = baseline.copy()
    lamb = np.zeros((n_split, 2))

    for i in range(n_split):
        main = np.random.rand(nobs) > 0.5
        rows1 = ~main & (treatment == 1)
        rows0 = ~main & (treatment == 0)

        mod1 = base.clone(model).fit(x.loc[rows1, :], (y.loc[rows1]))
        mod0 = base.clone(model).fit(x.loc[rows0, :], (y.loc[rows0]))

        B = mod0.predict(x)
        S = mod1.predict(x) - B
        baseline[:, i] = B
        cate[:, i] = S
        ES = S.mean()

        ## BLP
        # assume P(treat|x) = P(treat) = mean(treat)
        p = treatment.mean()
        reg_df = pd.DataFrame(
            dict(y=y, B=B, treatment=treatment, S=S, main=main, excess_S=S - ES)
        )
        reg = smf.ols(
            "y ~ B + I(treatment-p) + I((treatment-p)*(S-ES))", data=reg_df.loc[main, :]
        )
        reg_fit = reg.fit()
        blp[i, :] = reg_fit.params.iloc[2:4]
        blp_se[i, :] = get_treatment_se(reg_fit, cluster_id, main)[2:]

        lamb[i, 0] = reg_fit.params.iloc[-1] ** 2 * S.var()

        ## GATEs
        cutoffs = np.quantile(S, np.linspace(0, 1, n_group + 1))
        cutoffs[-1] += 1
        for k in range(n_group):
            reg_df[f"G{k}"] = (cutoffs[k] <= S) & (S < cutoffs[k + 1])

        g_form = "y ~ B + " + " + ".join(
            [f"I((treatment-p)*G{k})" for k in range(n_group)]
        )
        g_reg = smf.ols(g_form, data=reg_df.loc[main, :])
        g_fit = g_reg.fit()
        gate[i, :] = g_fit.params.values[2:]
        # g_fit.params.filter(regex="G").values
        gate_se[i, :] = get_treatment_se(g_fit, cluster_id, main)[2:]

        lamb[i, 1] = (gate[i, :] ** 2).sum() / n_group

    out = dict(
        gate=gate,
        gate_se=gate_se,
        blp=blp,
        blp_se=blp_se,
        Lambda=lamb,
        baseline=baseline,
        cate=cate,
        name=type(model).__name__,
    )
    return out
