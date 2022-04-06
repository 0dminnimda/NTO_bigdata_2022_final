import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def test():
    print(69)


def plot_the_loss_curve(epochs, error):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, error, label="Loss")
    plt.legend()
    plt.ylim([error.min()*0.94, error.max()*1.05])
    plt.show()


def plot_features_vs_label(features, label):
    for feature in features:
        plt.figure()
        # plt.xlabel("Epoch")
        # plt.ylabel("Root Mean Squared Error")

        plt.plot(feature, label)
        plt.legend()
        plt.ylim([label.min()*0.94, label.max()*1.05])
        plt.xlim([label.min()*0.94, label.max()*1.05])
        plt.show()


def pred_error(predicted, target):
    error = 1/len(predicted)
    summErr = 0
    for i in range(len(predicted)):
        summErr += abs(predicted[i]-target[i])/(predicted[i]+target[i])
    return error*summErr


def score(error):
    return 1000*(1-error)


def bucketize_column(column, min, max, resolution):
    boundaries = list(np.arange(int(min), int(max), resolution))
    return tf.feature_column.bucketized_column(column, boundaries)
