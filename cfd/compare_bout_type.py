# %% import libraries
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sqlite3
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
from utils import get_root_folder

# set up logging
logging.basicConfig(level=logging.INFO)
root_folder = get_root_folder()
(root_folder / "figs").mkdir(exist_ok=True)
# use custom matplotlib style
plt.style.use(root_folder / "main.mplstyle")

# %% read the results back from the database

db_file = root_folder / "results.db"
# read the results back from the database
with sqlite3.connect(db_file) as conn:
    df_tmp = pd.read_sql_query("SELECT * FROM results", conn)
conn.close()

df_tmp.sort_values(
    by=["dpf", "fish", "start_frame", "end_frame", "bout_type"], inplace=True
)
df_tmp

# %% plot relative error between the mean speed of CFD simulation and experiment
# (Fig. S6c)

fig, ax = plt.subplots(figsize=(1.5, 2))
sns.barplot(
    data=df_tmp,
    x="bout_type",
    y="RE (%)",
    fill=None,
    color="k",
    width=0.5,
    errorbar=None,
)
sns.stripplot(
    data=df_tmp,
    x="bout_type",
    y="RE (%)",
    hue="bout_type",
    palette=["#002fa7", "#800020"],
)
ax.set(xlabel="", ylabel="Relative error (%)")

# stats test on the relative error on bout types

# normality test
for bout_type in df_tmp["bout_type"].unique():
    data = df_tmp.query(f"bout_type == '{bout_type}'")["RE (%)"]
    _, p = shapiro(data)
    assert p > 0.05, "Data is not normally distributed"

# Extract the series values for each group
data = [group for _, group in df_tmp.groupby("bout_type")["RE (%)"]]
# Then pass the two series to the test function
_, p = ttest_ind(*data)

# add p-value to the plot

if p > 0.05:
    ax.text(
        0.5,
        0.9,
        "N.S.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
    )

fig.savefig(root_folder / "figs" / "bout_type_relative_error.svg")

# %% plot the efficiency of the bout types (Fig. S6d)

df_tmp2 = df_tmp.copy().dropna()
df_tmp2["ibamr_energy"] *= 100 # convert from erg to nano Joule
df_tmp2["cfd_distance"] *= 10 # convert from cm to mm
df_tmp2["exp_distance"] *= 10 # convert from cm to mm
df_tmp2["efficiency"] = df_tmp2["exp_distance"] / df_tmp2["ibamr_energy"]

fig, ax = plt.subplots(figsize=(1.5, 2))
sns.barplot(
    data=df_tmp2,
    x="bout_type",
    y="efficiency",
    fill=None,
    color="k",
    width=0.5,
    errorbar=None,
)
sns.stripplot(
    data=df_tmp2,
    x="bout_type",
    y="efficiency",
    hue="bout_type",
    palette=["#002fa7", "#800020"],
    ax=ax,
)
ax.set(xlabel="", ylabel="Efficiency (mm/nJ)")


# normality test
for bout_type in df_tmp["bout_type"].unique():
    data = df_tmp2.query(f"bout_type == '{bout_type}'")["efficiency"]
    _, p = shapiro(data)
    assert p > 0.05, "Data is not normally distributed"

# Extract the series values for each group
data = [group for _, group in df_tmp2.groupby("bout_type")["efficiency"]]
# Then pass the two series to the test function
_, p = ttest_ind(*data)
print(p)

# add p-value to the plot
if p > 0.05:
    ax.text(
        0.5,
        0.9,
        "N.S.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
    )

fig.savefig(root_folder / "figs" / "bout_type_exp_dist_efficiency.svg")
