import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

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

# Selecting features based on scatter plots and add a column of 1s for theta0
numerical_features = ['carlength', 'carwidth', 'carheight', 'highwaympg', 'horsepower']
numerical_data = car_data[numerical_features]
target_data = car_data[target_feature]
numerical_data.insert(0, 'X0', 1)
print(numerical_data.head())


# C. Split the dataset into training and testing sets 80% training, 20% testing
x_train = np.array(numerical_data.iloc[:164])
y_train = np.array(target_data.iloc[:164])
x_test = np.array(numerical_data.iloc[164:])
y_test = np.array(target_data.iloc[164:])
print("x_train", x_train)
print("y_train", y_train)

# create Array of thetas with 0 values as initial values
thetas = np.array([0, 0, 0, 0, 0, 0])
print("Array Of Thetas before Gradient Descent ", thetas)


# cost function J(theta)=1/2m * sum((theta0X0 + theta1X1 ....+ theta5X5)-y)^2
def cost_function(x_train, y_train, thetas):
    m = len(y_train)
    j = np.sum((x_train.dot(thetas) - y_train) ** 2) / (2 * m)
    return j


# Gradient Descent
# theta0 = theta0 - alpha * 1/m * sum((theta0X0 + theta1X1 ....+ theta5X5)-y)
def gradient_descent(x, y, thetas):
    m = len(y)
    alpha = 0.00000072
    iterations = 1000
    # array to store the cost function for every iteration initally all values are 0
    cost_history = [0] * iterations
    for iteration in range(iterations):
        # Hypothesis values for all X values h(x)=theta0X0 + theta1X1 ....+ theta5X5
        hypothesis = x.dot(thetas)
        # Difference between Hypothesis and Actual Y
        error = hypothesis - y
        # partial derivative of cost function with respect to theta0
        Partial_Drev = x.T.dot(error) / m
        # Gradient Descent
        thetas = thetas - alpha * Partial_Drev
        # calc the cost function to the iteration
        cost = cost_function(x, y, thetas)
        cost_history[iteration] = cost
    return thetas, cost_history


# start training the model
(thetas, all_costs) = gradient_descent(x_train, y_train, thetas)

#print Every X value and the Theta
print("Optimal Thetas  ")
print("Theta 0 ", thetas[0])
for i in range(1, len(thetas)-1):
    print(numerical_features[i-1], " ", thetas[i])
print(numerical_features[len(numerical_features)-1], " ", thetas[len(numerical_features)])

# plot the cost function
plt.plot(all_costs)
plt.xlabel("cost")
plt.ylabel("iterations")
plt.title("Cost Function")
plt.show()

# D. Test the model
y_pred = x_test.dot(thetas)
print("y_predected ", y_pred)
accuracy = 1 - np.mean(np.abs((y_pred - y_test) / y_test))
print("Accuracy ", accuracy)