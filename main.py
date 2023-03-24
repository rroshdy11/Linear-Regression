import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# A. Load the dataset
car_data = pd.read_csv("car_data.csv")

# B. Scatter plots to select features,  Select features
# Let's plot scatter plots between car price and all numerical features
features = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'highwaympg', 'horsepower', 'citympg']
target_feature = 'price'

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axs = axs.ravel()

for i, feature in enumerate(features):
    axs[i].scatter(car_data[feature], car_data[target_feature])
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel(target_feature)

plt.tight_layout()
plt.show()
# postive correlation: carwidth, curbweight, enginesize, horsepower
# negative correlation: highwaympg
numerical_features=['carlength', 'carwidth', 'curbweight', 'highwaympg', 'horsepower']

# C. Split the dataset into training and testing sets 80% training, 20% testing
x_train = car_data[numerical_features].iloc[:164]
y_train = car_data[target_feature].iloc[:164]
x_test = car_data[numerical_features].iloc[164:]
y_test = car_data[target_feature].iloc[164:]




