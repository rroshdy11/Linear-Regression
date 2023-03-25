import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

def plottingFeature(car_data):
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

def eqnLinear(car_data, target_feature):
    # Equation of Multiple Linear Regression Model y = theta0X0 + theta1x1 + theta2x2 + theta3x3 + theta4x4 + theta5x5
    numerical_features = ['carlength', 'carwidth', 'carheight', 'highwaympg', 'horsepower']
    numerical_data = car_data[numerical_features]
    target_data = car_data[target_feature]
    numerical_data.insert(0, 'X0', 1)
    print(numerical_data.head())
    return numerical_data, target_data , numerical_features

def splitData(numerical_data, target_data):
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
    return x_train, y_train, x_test, y_test, thetas


# cost function J(theta)=1/2m * sum((theta0X0 + theta1X1 ....+ theta5X5)-y)^2
def cost_function(x_train, y_train, thetas):
    m = len(y_train)
    j = np.sum((x_train.dot(thetas) - y_train) ** 2) / (2 * m)
    return j


# Gradient Descent

def gradient_descent(x, y, thetas):
    m = len(y)
    alpha = 0.00000072
    iterations = 1000
    cost_history = [0] * iterations
    for iteration in range(iterations):
        hypothesis = x.dot(thetas)
        error = hypothesis - y
        gradient = x.T.dot(error) / m
        thetas = thetas - alpha * gradient
        cost = cost_function(x, y, thetas)
        cost_history[iteration] = cost
    return thetas, cost_history

def training(x_train, y_train, thetas):
    # start training the model
    (thetas, all_costs) = gradient_descent(x_train, y_train, thetas)
    return thetas,all_costs

def printXThetas(numerical_features, thetas):
    #print Every X value and the Theta
    print("Optimal Thetas  ")
    print("Theta 0 ", thetas[0])
    for i in range(1, len(thetas)):
        print(numerical_features[i-1], " ", thetas[i])

def plotCostFunction(all_costs):
    # plot the cost function
    plt.plot(all_costs)
    plt.xlabel("cost")
    plt.ylabel("iterations")
    plt.title("Cost Function")
    plt.show()

def test(thetas, x_test, y_test):
    # D. Test the model
    y_pred = x_test.dot(thetas)
    print("y_predected ", y_pred)
    accuracy = 1 - np.mean(np.abs((y_pred - y_test) / y_test))
    print("Accuracy ", accuracy)

if __name__=="__main__":
    # A. Load the dataset
    car_data = pd.read_csv("car_data.csv")
    plottingFeature(car_data)
    numerical_data, target_data, numerical_features = eqnLinear(car_data, 'price')
    x_train, y_train, x_test, y_test, thetas = splitData(numerical_data, target_data)
    thetas,all_costs=training(x_train, y_train, thetas)
    printXThetas(numerical_features, thetas)
    plotCostFunction(all_costs)
    test(thetas, x_test, y_test)
