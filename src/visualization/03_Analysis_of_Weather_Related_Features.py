# --------------------------------------------------------------
# Importing Libraries
# --------------------------------------------------------------
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("..")
import visualization.plot_settings

# --------------------------------------------------------------
# Loading Data
# --------------------------------------------------------------

preprocessed_data = pd.read_pickle("../../data/interim/01_preprocessed_data.plk")

# --------------------------------------------------------------
# Analysis of Weather-Related Features
# --------------------------------------------------------------
"""
the first and most common way to measure the relationship 
between two different continuous variables is to measure their correlation.
"""
# correlation coefficient assuming there's a linear relatioship between random variable
#  define a function that performs the analysis between the variables


def plot_correlations(data, col):
    # get correlation betwwen col and registered rides
    corr_reg = np.corrcoef(data[col], data["registered"])[0, 1]
    ax = sns.regplot(
        x=col,
        y="registered",
        data=data,
        scatter_kws={"alpha": 0.1},
        label=f"Registered rides (correlation: {corr_reg: 0.3f})",
    )
    # get correlation betwwen col and registered rides
    corr_casu = np.corrcoef(data[col], data["casual"])[0, 1]
    ax = sns.regplot(
        x=col,
        y="casual",
        data=data,
        scatter_kws={"alpha": 0.1},
        label=f"Casual rides (correlation: {corr_reg: 0.3f})",
    )
    # adjust legand alpha
    legend = ax.legend()
    for lh in legend.legendHandles:
        lh.set_alpha(0.5)
        ax.set_ylabel("rides")
        ax.set_title(f"Correlation between rides and {col}")
        return ax


# Applying the previously defined function to the four columns (temp, atemp, hum, and windspeed)
plt.figure()
ax = plot_correlations(preprocessed_data, "temp")
plt.savefig(
    "../../reports/Figs/0_9_Correlation between rides and temp.png", format="png"
)
plt.figure()
ax = plot_correlations(preprocessed_data, "atemp")
plt.savefig(
    "../../reports/Figs/10_Correlation between rides and atemp.png", format="png"
)
plt.figure()
ax = plot_correlations(preprocessed_data, "hum")
plt.savefig("../../reports/Figs/11_Correlation between rides and hum.png", format="png")
plt.figure()
ax = plot_correlations(preprocessed_data, "windspeed")
plt.savefig(
    "../../reports/Figs/12_Correlation between rides and windspeed.png",
    format="png",
)
""" 
A big problem with the correlation coefficient is that it assumes the relationship between 
two variables is always linear. But in reality, most relationships in nature aren't linear.
"""
# --------------------------------------------------------------
# Spearman rank correlation for monotonic relatioship between random variable
# Evaluating the Difference between the Pearson and Spearman Correlations
# --------------------------------------------------------------

# define random variables
x = np.linspace(0, 5, 100)
y_lin = 0.5 * x + 0.1 * np.random.randn(100)
y_mon = np.exp(x) + 0.1 * np.random.randn(100)

# compute correlations
from scipy.stats import pearsonr, spearmanr

corr_lin_pearson = pearsonr(x, y_lin)[0]
corr_lin_spearman = spearmanr(x, y_lin)[0]

corr_mon_pearson = pearsonr(x, y_mon)[0]
corr_mon_spearman = spearmanr(x, y_mon)[0]

# visualize variables
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(x, y_lin)
ax1.set_title(
    f"Linear relationship\n \
Pearson: {corr_lin_pearson:.3f}, \
Spearman: {corr_lin_spearman:.3f}"
)
ax2.scatter(x, y_mon)
ax2.set_title(
    f"Monotonic relationship\n \
Pearson: {corr_mon_pearson:.3f}, \
Spearman: {corr_mon_spearman:.3f}"
)
plt.savefig(
    "../../reports/Figs/13_Difference between the Pearson and Spearman Correlations.png",
    format="png",
)

""" 
when the relationship between the two variables is linear (the figure on the left), 
the two correlation coefficients are very similar. 

In the monotonic relationship (the figure on the right), the linear assumption of 
the Pearson correlation fails, and, although the correlation coefficient is still quite high (0.856),
it is not capable of capturing the perfect relationship between the two variables. 

On the other hand, the Spearman correlation coefficient is 1, which means that it succeeds in capturing 
the almost perfect relationship between the two variables.

"""


# define function for computing correlations
def compute_correlations(data, col):
    pearson_regis = pearsonr(data[col], data["registered"])[0]
    pearson_casul = pearsonr(data[col], data["casual"])[0]
    spearman_regis = spearmanr(data[col], data["registered"])[0]
    spearman_casul = spearmanr(data[col], data["casual"])[0]
    return pd.Series(
        {
            "Pearson (Registered)": pearson_regis,
            "Pearson (Casual)": pearson_casul,
            "Spearman (Registered)": spearman_regis,
            "Spearman (Casual)": spearman_casul,
        }
    )


"""
Note: function returns a pandas.Series() object, 
which will be used to create a new dataset containing the different correlations
"""
# compute correlation measures between different features
cols = ["temp", "atemp", "hum", "windspeed"]
corr_data = pd.DataFrame(
    index=[
        "Pearson (Registered)",
        "Spearman (Registered)",
        "Pearson (Casual)",
        "Spearman (Casual)",
    ]
)
for col in cols:
    corr_data[col] = compute_correlations(preprocessed_data, col)
corr_data.T

# plot correlation matrix
cols = ["temp", "atemp", "hum", "windspeed", "registered", "casual"]
plot_data = preprocessed_data[cols]
corr = plot_data.corr()
fig = plt.figure(figsize=(10, 8))
plt.matshow(corr, fignum=fig.number)
plt.xticks(range(len(plot_data.columns)), plot_data.columns)
plt.yticks(range(len(plot_data.columns)), plot_data.columns)
plt.colorbar()
plt.ylim([5.5, -0.5])
plt.title("Matrix correlations.png")
plt.savefig("../../reports/Figs/14_matrix correlations.png", format="png")
