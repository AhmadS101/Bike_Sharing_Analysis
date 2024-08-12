# --------------------------------------------------------------
# Importing Libraries
# --------------------------------------------------------------

import pandas as pd

# --------------------------------------------------------------
# Loading Data
# --------------------------------------------------------------
hourly_data = pd.read_csv("../../data/raw/hour.csv")

# --------------------------------------------------------------
# print some generic statistics about the data
# --------------------------------------------------------------

print(f"Shape of data: {hourly_data.shape}")
"""
using two sum() below 
the first for find null in each column and 
the second for sum all columns null values
"""
print(f"Number of missing valuse in the data: {hourly_data.isnull().sum().sum()}")
hourly_data.describe().T  # "T" to show statical as columns

# --------------------------------------------------------------
# Data Preprocessing
# --------------------------------------------------------------
# 1.01: Preprocessing Temporal and Weather Features

preprocessed_data = hourly_data.copy()  # copying original data
# transform seasons
seasons_mapping = {1: "winter", 2: "spring", 3: "summer", 4: "fall"}
preprocessed_data["season"] = preprocessed_data["season"].apply(
    lambda x: seasons_mapping[x]
)

# transform yr
yr_mapping = {0: 2011, 1: 2012}
preprocessed_data["yr"] = preprocessed_data["yr"].apply(lambda x: yr_mapping[x])

# transform weekday
weekday_mapping = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}
preprocessed_data["weekday"] = preprocessed_data["weekday"].apply(
    lambda x: weekday_mapping[x]
)

# transform weathersit
weather_mapping = {
    1: "clear",
    2: "cloudy",
    3: "light_rain_snow",
    4: "heavy_rain_snow",
}
preprocessed_data["weathersit"] = preprocessed_data["weathersit"].apply(
    lambda x: weather_mapping[x]
)

# transform hum and windspeed
preprocessed_data["hum"] = preprocessed_data["hum"] * 100
preprocessed_data["windspeed"] = preprocessed_data["windspeed"] * 67

# visualize preprocessed columns
cols = ["season", "yr", "weekday", "weathersit", "hum", "windspeed"]
preprocessed_data[cols].sample(10, random_state=123)


preprocessed_data.to_pickle("../../data/interim/01_preprocessed_data.plk")
