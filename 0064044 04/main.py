import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_data():
    data = np.genfromtxt("hw04_data_set.csv", delimiter=",", skip_header=1)
    X = np.array(data[:, 0])
    Y = np.array(data[:, 1])
    train_x = np.array(X[0:100])
    train_y = np.array(Y[0:100])
    test_x = np.array(X[100:])
    test_y = np.array(Y[100:])
    N = data.shape[0]
    return train_x, train_y, test_x, test_y, N

def my_mean(arr):
    if arr.size == 0:
        return 0
    else: return np.mean(arr)

def plot_estimate(train_x, train_y, test_x, test_y, h):
    minimum_value = 0
    maximum_value = np.max(train_x)
    bin_width = h
    left_borders = np.arange(minimum_value, maximum_value, bin_width)
    right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)
    p_hat = np.asarray(
        [my_mean(train_y[(left_borders[b] < train_x) & (train_x <= right_borders[b])]) for b in range(len(left_borders))])

    plt.figure(figsize=(10, 6))
    plt.plot(train_x[:], train_y[:], "b.", markersize=10)
    plt.plot(test_x[:], test_y[:], "r.", markersize=10)
    for b in range(len(left_borders)):
        plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
    for b in range(len(left_borders) - 1):
        plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")
    plt.title("Regressogram h=3")
    plt.xlabel("x")
    plt.ylabel("y")
    return p_hat


def plot_estimate_RMS(train_x, train_y, test_x, test_y, h):
    bin_width = h
    minimum_value = 0
    maximum_value = np.max(train_x)+bin_width/2
    data_interval = np.linspace(minimum_value, maximum_value, 1601)

    p_hat = np.asarray(
        [my_mean(train_y[np.abs(data_interval[r] - train_x) <= bin_width/2]) for r in range(len(data_interval))])

    plt.figure(figsize=(10, 6))
    plt.plot(train_x[:], train_y[:], "b.", markersize=10)
    plt.plot(test_x[:], test_y[:], "r.", markersize=10)
    plt.plot(data_interval, p_hat, "k-")
    plt.title(" Running Mean Smoother h=3")
    plt.xlabel("x")
    plt.ylabel("y")
    return data_interval , p_hat

def Kernel(bin_distance):
    return 1 / np.sqrt(2 * np.math.pi) * np.exp(-0.5 * (bin_distance ** 2))


def plot_estimate_KS(train_x, train_y, test_x, test_y, h):
    bin_width = h
    minimum_value = 0
    maximum_value = 60
    data_interval = np.linspace(minimum_value, maximum_value, 1601)

    p_hat = np.asarray([np.sum(Kernel((x - train_x) / bin_width)* train_y)/np.sum(Kernel((x - train_x) / bin_width)) for x in data_interval])
    plt.figure(figsize=(10, 6))
    plt.plot(train_x[:], train_y[:], "b.", markersize=10)
    plt.plot(test_x[:], test_y[:], "r.", markersize=10)
    plt.plot(data_interval, p_hat, "k-")
    plt.title("Kernel Smoother h=1")
    plt.xlabel("x")
    plt.ylabel("y")
    return data_interval , p_hat

def calculate_error_KS(Y,X,bin_width, test_x, test_y):
    error = np.sqrt(
        np.sum(
            np.square(test_y - [np.sum(Kernel((test_x[x] - X)/ bin_width)* Y)/np.sum(Kernel((test_x[x] - X) / bin_width)) for x in range(test_x.shape[0])])) /
        test_x.shape[
            0])
    print("Kernel Smoother error is: ", error)
    return error

def calculate_error_RMS(Y,X,bin_width, test_x, test_y):
    error = np.sqrt(
        np.sum(np.square(test_y - [my_mean(Y[np.abs(X - test_x[x]) <= bin_width/2]) for x in range(test_x.shape[0])])) / test_x.shape[
            0])
    print("Running Mean Smoother error is: " , error)
    return error
def calculate_error(y_hat, test_x, test_y):
    error = np.sqrt(
        np.sum(np.square(test_y - [y_hat[(test_x[x] / 3).astype(int)] for x in range(test_x.shape[0])])) / test_x.shape[
            0])
    print("Regressogram error is: ", error)
    return error

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, N = parse_data()
    y_hat = plot_estimate(train_x, train_y, test_x, test_y, 3)
    interval, y_hat2 = plot_estimate_RMS(train_x, train_y, test_x, test_y, 3)
    interval2, y_hat3 = plot_estimate_KS(train_x, train_y, test_x, test_y, 1)

    error = calculate_error(y_hat, test_x, test_y)
    error2 = calculate_error_RMS(train_y, train_x, 3, test_x, test_y)
    error3 = calculate_error_KS(train_y, train_x, 1, test_x, test_y)

    plt.show()
