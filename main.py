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
numerical_features = ['carlength', 'carwidth', 'curbweight', 'highwaympg', 'horsepower']
numerical_data = car_data[numerical_features]
target_data = car_data[target_feature]
print(numerical_data.head())
numerical_data.insert(0, 'x0', 1)

# C. Split the dataset into training and testing sets 80% training, 20% testing
x_train = numerical_data[:164]
y_train = target_data[:164]
x_test = numerical_data[164:]
y_test = target_data[164:]
# transform the data into numpy arrays
X_train = np.array(x_train)
Y_train = np.array(y_train)
X_test = np.array(x_test)
Y_test = np.array(y_test)
# create Array of thetas
thetas = np.array([0, 0, 0, 0, 0, 0])
print("Array Of Thetas ", thetas)




# Gradient Descent

def gradient_descent(x, y, thetas):
    m = len(y)
    alpha = 0.0001
    iterations = 1000
    J_history = [0] * iterations
    for iteration in range(iterations):
        hypothesis = x.dot(thetas)
        error = hypothesis - y
        gradient = x.T.dot(error) / m
        thetas = thetas - alpha * gradient
        J_history[iteration] = np.sum(error ** 2) / (2 * m)
    return thetas, J_history


# start training the model
(thetas, all_costs) = gradient_descent(X_train, Y_train, thetas)
for i in range(len(thetas)):
    print( " ", thetas[i])

# plot the cost function
plt.plot(all_costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.show()

