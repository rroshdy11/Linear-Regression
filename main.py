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

# postive correlation: carwidth, curbweight, enginesize, horsepower
# negative correlation: highwaympg
# Equation of Multiple Linear Regression Model y = theta0X0 + theta1x1 + theta2x2 + theta3x3 + theta4x4 + theta5x5
numerical_features = ['carlength', 'carwidth', 'carheight', 'highwaympg', 'horsepower']
numerical_data = car_data[numerical_features]
target_data = car_data[target_feature]
print(numerical_data.head())
# MinMax Scaling
X_min = np.min(numerical_data, axis=0)
X_max = np.max(numerical_data, axis=0)
numerical_data = (numerical_data - X_min) / (X_max - X_min)

numerical_data.insert(0, 'X0', 1)
print("After MinMax Scaling", numerical_data)
# C. Split the dataset into training and testing sets 80% training, 20% testing
x_train = numerical_data[:164]
y_train = target_data[:164]
x_test = numerical_data[164:]
y_test = target_data[164:]
print("x_train", x_train)
print("y_train", y_train)
# transform the data into numpy arrays
X_train = np.array(x_train)
Y_train = np.array(y_train)
X_test = np.array(x_test)
Y_test = np.array(y_test)
# create Array of thetas
thetas = np.array([0, 0, 0, 0, 0, 0])
print("Array Of Thetas ", thetas)


# cost function J(theta)=1/2m * sum((theta0X0 + theta1X1 ....+ theta5X5)-y)^2
def cost_function(x_train, y_train, thetas):
    m = len(y_train)
    j = np.sum((x_train.dot(thetas) - y_train) ** 2) / (2 * m)
    return j


# Gradient Descent

def gradient_descent(x, y, thetas):
    m = len(y)
    alpha = 0.001
    iterations = 2000
    cost_history = [0] * iterations
    for iteration in range(iterations):
        hypothesis = x.dot(thetas)
        error = hypothesis - y
        gradient = x.T.dot(error) / m
        thetas = thetas - alpha * gradient
        cost = cost_function(x, y, thetas)
        cost_history[iteration] = cost
    return thetas, cost_history


# start training the model
(thetas, all_costs) = gradient_descent(X_train, Y_train, thetas)
for i in range(len(thetas)):
    print('Theta ', i, " ", thetas[i])

# plot the cost function
plt.plot(all_costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.show()

