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
#  Time Series Analysis
# --------------------------------------------------------------
"""
When studying time series, an important concept is the notion of stationarity. A time 
series is said to be strongly stationary if all aspects of its behavior do not change in 
time. 


we can rely on two different techniques for identifying time series stationarity:
rolling statistics and augmented Dickey-Fuller stationarity test

define function for plotting rolling statistics and 
Augmented Dickey-Fuller stationarity test (ADF)) for time series.
"""
from statsmodels.tsa.stattools import adfuller


def test_stationarity(ts, window=10, center=True, **kwargs):
    # create a dataframe for plotting
    plot_data = pd.DataFrame(ts)
    plot_data["rolling_mean"] = ts.rolling(window=window, center=center).mean()
    plot_data["rolling_std"] = ts.rolling(window=window, center=center).std()
    # compute p-value of Dickey-Fuller test
    p_value = adfuller(ts)[1]
    # Plotting the data
    ax = plot_data.plot(**kwargs)
    ax.set_title(f"Dickey-Fuller p-value: {p_value:.3f}")
    plt.show()


# Example usage with DataFrame 'daily_rides'
daily_rides = preprocessed_data[["dteday", "registered", "casual"]]
daily_rides = daily_rides.groupby("dteday").sum()

# convert index to DateTime object
daily_rides.index = pd.to_datetime(daily_rides.index)
plt.figure()
# Test stationarity on the 'registered' series
test_stationarity(
    ts=daily_rides["registered"], window=10, center=True, ylabel="Rides", xlabel="Date"
)
plt.savefig("../../reports/Figs/15_daily_registered_original.png", format="png")
test_stationarity(
    ts=daily_rides["casual"], window=10, center=True, ylabel="Rides", xlabel="Date"
)
plt.savefig("../../reports/Figs/16_daily_registered_original.png", format="png")

""" 
From the performed tests, we can see that neither the moving average nor standard 
deviations are stationary. Furthermore, the Dickey-Fuller test returns values of
0.355 and 0.372 for the registered and casual columns, respectively.
This is strong evidence that the time series is not stationary, and we need to
process them in order to obtain a stationary one.


A common way to detrend a time series and make it stationary is to subtract either its 
rolling mean or its last value, or to decompose it into a component that will contain its 
trend, seasonality, and residual components.
"""
# substract rolling mean
registered = daily_rides["registered"]
registered_ma = registered.rolling(window=10).mean()
registered_ma_differance = registered - registered_ma
registered_ma_differance.dropna(inplace=True)
# plot tested stationarity for registered
plt.figure()
test_stationarity(registered_ma_differance, figsize=(20, 5))
plt.savefig(
    "../../reports/Figs/17_tested_stationarity_for_registered.png", format="png"
)

# plot tested stationarity for registered
casual = daily_rides["casual"]
casual_ma = casual.rolling(window=10).mean()
casual_ma_differance = casual - casual_ma
casual_ma_differance.dropna(inplace=True)
plt.figure()
test_stationarity(casual_ma_differance, figsize=(20, 5))
plt.savefig("../../reports/Figs/18_tested_stationarity_for_casual.png", format="png")


# subtract last value
registered = daily_rides["registered"]
registered_differance = registered - registered.shift()
registered_differance.dropna(inplace=True)
# plot tested stationarity for registered
plt.figure()
test_stationarity(registered_differance, figsize=(20, 5))
plt.savefig(
    "../../reports/Figs/19_tested_stationarity_for_registered_as_lastValue.png",
    format="png",
)

# plot tested stationarity for casual
casual = daily_rides["casual"]
casual_differance = casual - casual.shift()
casual_differance.dropna(inplace=True)
plt.figure()
test_stationarity(casual_ma_differance, figsize=(20, 5))
plt.savefig(
    "../../reports/Figs/20_tested_stationarity_for_casual_as_lastValue.png",
    format="png",
)
"""
both of the techniques returned a time series, which is stationary, according to the Dickey-Fuller test. 
Note that an interesting pattern occurs in the casual series: a rolling standard deviation exhibits 
a clustering effect, that is, periods in which the standard deviation is higher and periods 
in which it is lower.
"""

# --------------------------------------------------------------
# decompose the number of rides into three separate components, trend, seasonal, and residual components
# --------------------------------------------------------------

from statsmodels.tsa.seasonal import seasonal_decompose

registered_decomposition = seasonal_decompose(daily_rides["registered"])
casual_decomposition = seasonal_decompose(daily_rides["casual"])

# plot decompositions
registered_plot = registered_decomposition.plot()
registered_plot.set_size_inches(10, 8)

casual_plot = casual_decomposition.plot()
casual_plot.set_size_inches(10, 8)

registered_plot.savefig(
    "../../reports/Figs/21_registered_decomposition.png", format="png"
)
casual_plot.savefig("../../reports/Figs/22_casual_decomposition.png", format="png")

# test residuals for stationarity
plt.figure()
test_stationarity(registered_decomposition.resid.dropna(), figsize=(25, 5))
plt.savefig("../../reports/Figs/23_registered_resid.png", format="png")

plt.figure()
test_stationarity(registered_decomposition.resid.dropna(), figsize=(20, 5))
plt.savefig("../../reports/Figs/24_registered_resid.png", format="png")
