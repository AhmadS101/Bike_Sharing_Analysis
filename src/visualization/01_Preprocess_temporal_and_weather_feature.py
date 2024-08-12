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
#  Registered versus Casual Use Analysis
# --------------------------------------------------------------
"""
From our data, we need to determine the impact of temperature and weather
on bike-sharing rides. We have two types of rides: registered and casual, so let's start.
"""

# assert that total number of rides is equal to the sum of registered and casual ones
assert (
    preprocessed_data.casual + preprocessed_data.registered == preprocessed_data.cnt
).all(), " Sum of casual and registered rides not equal to total number of rides"


# plot distributions of registered vs casual rides
sns.distplot(preprocessed_data["registered"], label="registered")
sns.distplot(preprocessed_data["casual"], label="casual")
plt.legend()
plt.title("Rides distributions")
plt.xlabel("rides")
plt.savefig("../../reports/Figs/01_rides_distributions.png", format="png")

"""
We can see that registered users take many more rides than casual ones
"""


# plot evolution of rides over time
plot_data = preprocessed_data[["registered", "casual", "dteday"]]
ax = plot_data.groupby("dteday").sum().plot(figsize=(15, 5))
ax.set_xlabel("time")
ax.set_ylabel("number of rides per day")
plt.savefig("../../reports/Figs/02_rides_daily.png", format="png")

"""
The number of registered rides is always above and significantly higher than 
the number of casual rides per day. Furthermore, we can observe that during winter, 
the overall number of rides decreases (due to bad weather and low temperatures have
a negative impact on ride sharing services).
"""


# smothing the data with rolling statistics
"""
Create new dataframe with necessary for plotting columns, and obtain 
number of rides per day, by grouping over each day. the define window 
for computing the rolling mean and standard deviation.
"""

plot_data = preprocessed_data[["registered", "casual", "dteday"]]
plot_data = plot_data.groupby("dteday").sum()
window = 7
rolling_means = plot_data.rolling(window).mean()
rolling_deviations = plot_data.rolling(window).std()

"""
Create a plot of the series, where we first plot the series of rolling 
means, then we color the zone between the series of rolling means +- 2 
rolling standard deviations
"""

ax = rolling_means.plot()
ax.fill_between(
    rolling_means.index,
    rolling_means["registered"] + 2 * rolling_deviations["registered"],
    rolling_means["registered"] - 2 * rolling_deviations["registered"],
    alpha=0.2,
)
ax.fill_between(
    rolling_means.index,
    rolling_means["casual"] + 2 * rolling_deviations["casual"],
    rolling_means["casual"] - 2 * rolling_deviations["registered"],
    alpha=0.2,
)
ax.set_xlabel("time")
ax.set_ylabel("number of rides per day")
plt.savefig("../../reports/Figs/03_rides_aggregated.png.png", format="png")


# the distributions of the requests over separate hours and days of the week.
# select relevant columns
plot_data = preprocessed_data[["hr", "weekday", "registered", "casual"]]

"""
transform the data into a format, in number of entries are computed as 
count,for each distinct hr, weekday and type (registered or casual)
"""

plot_data = plot_data.melt(
    id_vars=["hr", "weekday"], var_name="type", value_name="count"
)
"""
create FacetGrid object, in which a grid plot is produced.As columns, 
we have the various days of the week,as rows, the different types (registered and casual)
"""
grid = sns.FacetGrid(
    plot_data,
    row="weekday",
    col="type",
    height=2.5,
    aspect=2.5,
    row_order=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)
# populate the FacetGrid with the specific plots
grid.map(sns.barplot, "hr", "count", alpha=0.5)
grid.savefig("../../reports/Figs/04_weekday_hour_distributions.png", format="png")
"""
the highest number of rides for registered users takes place around 8 AM and at 6 PM. 
This is totally in line with our expectations, as it is likely that most registered users
use the bike sharing service for commuting.
"""

# --------------------------------------------------------------
#  Analyzing Seasonal Impact on Rides
# --------------------------------------------------------------
# we'll investigate the impact of the different season on the total number of rides
# select subset of data contains season with hr columns
plot_data = preprocessed_data[["hr", "season", "registered", "casual"]]

# unpivot data from wide to long format
plot_data = plot_data.melt(
    id_vars=["hr", "season"], var_name="type", value_name="count"
)

# define FacetGrid
grid = sns.FacetGrid(
    plot_data,
    row="season",
    col="type",
    height=2.5,
    aspect=2.5,
    row_order=["winter", "spring", "summer", "fall"],
)
# apply plotting function to each element in the grid
grid.map(sns.barplot, "hr", "count", alpha=0.5)
plt.show()
grid.savefig("../../reports/Figs/0_5_season_impact.png", format="png")


# select subset of data contains season with weekday columns
plot_data = preprocessed_data[["weekday", "season", "registered", "casual"]]
plot_data = plot_data.melt(
    id_vars=["weekday", "season"], var_name="type", value_name="count"
)

grid = sns.FacetGrid(
    plot_data,
    row="season",
    col="type",
    height=2.5,
    aspect=2.5,
    row_order=["winter", "spring", "summer", "fall"],
)

grid.map(
    sns.barplot,
    "weekday",
    "count",
    alpha=0.5,
    order=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)

grid.savefig("../../reports/Figs/0_6_season_impact_to_weekday.png", format="png")

"""
Analyzing Seasonal Impact on Rides. There is a decreasing number of registered rides over 
the weekend (compared to the rest of the week), while the number of casual rides increases.
"""
