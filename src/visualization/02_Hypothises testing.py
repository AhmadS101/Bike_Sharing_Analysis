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
#  Hypothesis Tests
# --------------------------------------------------------------
"""
Based on the earlier work of preparing time and weather data, we noticed that registered users take 
more rides than casual users. Now, we’ll test the ideas we have about how temperature affects these rides.

Hypothesis Tests: part of the statistics field in which a general conclusion can be done about a large group 
(a population) based on the analysis and measurements performed on a smaller group (a sample).

Note:  
that this value will serve in formulating the null hypothesis because, here, you are explicitly 
computing the population statistic—that is, the average number of rides.
"""

# Estimating Average Registered Rides
# compute population mean of registered rides

population_mean = preprocessed_data.registered.mean()

"""
there are two example we'll perform the first for estemating the true avarage numbers of rides perform by 
registerd for a SUMMER then, we'll perform the same ture avarge but using random sample which a true representation 
of the population not a summer only
"""
# get sample of the data (summer 2011)
sample = preprocessed_data[
    (preprocessed_data["season"] == "summer") & (preprocessed_data["yr"] == 2011)
].registered

# perform t_test and p-value, the significance level is 0.05
from scipy.stats import ttest_1samp

test_result = ttest_1samp(sample, population_mean)
print(f"Test statistic: {test_result[0]:0.03f}, p-value: {test_result[1]: 0.03f}")

""" 
The result of the previous test returns a p-value smaller than 0.001, which is less than 
the predefined critical value. Therefore, you can reject the null hypothesis and assume that 
the alternative hypothesis is correct.


we computed the average number of rides on the true population; therefore, the value computed 
by the statistical test should be the same. So why have you rejected the null hypothesis?

sample is not a true representation of the population, but rather a biased one. In fact, you 
selected only entries from the summer of 2011. Therefore, neither data from the 
full year is present, nor entries from 2012. 
"""

# get sample as 5% of the full data
import random

random.seed(111)
sample_unbiased = preprocessed_data.registered.sample(frac=0.5)
test_result_unbiased = ttest_1samp(sample_unbiased, population_mean)
print(
    f"Test statistic: {test_result_unbiased[0]:0.03f}, p-value: {test_result_unbiased[1]: 0.03f}"
)
"""
This time, the computed p-value is equal to 0.685, which is much larger than the critical 0.05, 
and so, you cannot reject the null hypothesis.we saw the importance of having an unbiased sample 
of the data, as test results can be easily compromised if working with biased data.
"""

# --------------------------------------------------------------
# Hypothesis Testing on Registered Rides
# --------------------------------------------------------------
""" 
H_0 : average registered rides over weekdays-average registered rides over 
weekend=0 
H_a : average registered rides over weekdays-average registered rides over 
weekend≠0
"""

# define mask, indicating if the day is weekend or work day
weekend_days = ["Saturday", "Sunday"]
weekend_mask = preprocessed_data.weekday.isin(weekend_days)
workingdays_mask = ~preprocessed_data.weekday.isin(weekend_days)

# select registered rides for the weekend and working days
weekend_data = preprocessed_data.registered[weekend_mask]
workingdays_data = preprocessed_data.registered[workingdays_mask]

# perform ttest
from scipy.stats import ttest_ind

"""
ttest_ind function from the scipy.stats module is used to
perform a T-test for the means of two independent samples.
"""
test_result = ttest_ind(weekend_data, workingdays_data)
print(f"Test statistic: {test_result[0]:0.03f}, p-value: {test_result[1]: 0.03f}")
""" 
The resulting p-value from this test is less than 0.0001, which is far below the 
standard critical 0.05 value. As a conclusion, we can reject the null hypothesis 
and confirm that our initial observation is correct
"""

#  plot distributions of registered rides for working vs weekend days
sns.distplot(weekend_data, label="weekend days")
sns.distplot(workingdays_data, label="working days")
plt.legend()
plt.xlabel("rides")
plt.ylabel("frequency")
plt.title("Registered rides distributions")
plt.savefig("../../reports/Figs/0_7_Registered rides distributions.png", format="png")

""" 
the second assumption from the last section that is, casual users perform more
rides during the weekend. In this case, the null hypothesis is that the average 
number of rides during working days is the same as the average number of rides 
during the weekend, both performed only by casual customers.
"""
# select casual rides for the weekend and working days
# we'll using the previouse mask
weekend_data = preprocessed_data.casual[weekend_mask]
workingdays_data = preprocessed_data.casual[workingdays_mask]

# perform ttest
test_result = ttest_ind(weekend_data, workingdays_data)
print(f"Test statistic: {test_result[0]:0.03f}, p-value: {test_result[1]: 0.03f}")

# plot distribution of casual rides for working days VS weekend days
sns.distplot(weekend_data, label="weekend days")
sns.distplot(workingdays_data, label="working days")
plt.legend()
plt.xlabel("rides")
plt.ylabel("frequancy")
plt.title("Casual rides distributions")
plt.savefig("../../reports/Figs/0_8_casual rides distributions.png", format="png")

"""
The p-value returned from the previous code snippet is 0, which is strong 
evidence against the null hypothesis.

In conclusion:
we can say that there is a statistically significant difference between  
the number of rides on working days and weekend days for both casual and  
registered customers. 
"""
