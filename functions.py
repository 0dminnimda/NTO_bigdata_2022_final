import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow import feature_column

tf.keras.backend.set_floatx('float32')


def test():
    print(69)


def plot_the_loss_curve(epochs, error):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Error")

    plt.plot(epochs, error, label="Loss")
    plt.legend()
    plt.ylim([error.min()*0.94, error.max()*1.05])
    plt.show()


def plot_the_loss_curves(epochs, mae_training, mae_validation):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Error")

    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
    plt.legend()

    # We're not going to plot the first epoch, since the loss on the first epoch
    # is often substantially greater than the loss for other epochs.
    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)

    plt.ylim([bottom_of_y_axis, top_of_y_axis])
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


def tf_error(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred-y_true)/(y_pred+y_true))


class CustomAccuracy(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf_error(y_true, y_pred)
        # return tf.reduce_mean(tf.abs(y_pred-y_true)/(y_pred+y_true))
        # return error*summErr
        # mse = tf.reduce_mean(tf.square(y_pred-y_true))
        # rmse = tf.math.sqrt(mse)
        # return rmse / tf.reduce_mean(tf.square(y_true)) - 1


def score(error):
    return 1000*(1-error)


def bucketize_column(column, min, max, resolution):
    # boundaries = list(np.arange(int(min), int(max), resolution))
    boundaries = list(np.linspace((min), (max), resolution))
    return tf.feature_column.bucketized_column(column, boundaries)


def create_model(my_learning_rate, feature_layer):
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(feature_layer)

    # Add one linear layer to the model to yield a simple linear regressor.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Construct the layers into a model that TensorFlow can execute.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss=CustomAccuracy(),
                  metrics=[tf_error])

    return model


def train_model(model, features, label, my_epochs,
                my_batch_size=None, my_validation_split=0.1):
    history = model.fit(
        x={name: np.array(value) for name, value in features.items()},
        y=np.array(label),
        batch_size=my_batch_size,
        epochs=my_epochs,
        validation_split=my_validation_split,
        shuffle=True)

    # Gather the model's trained weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    # print(dir(hist), hist.columns)
    rmse = hist["tf_error"]

    return epochs, rmse, history.history


def evaluate_model(model, features, label, my_batch_size=None):
    return model.evaluate(
        x={name: np.array(value) for name, value in features.items()},
        y=np.array(label),
        batch_size=my_batch_size)


def replace_with_linear_interpolation(train, indices):
    train['rownum'] = np.arange(train.shape[0])
    invalid = train.drop(train.loc[indices].index)
    f = interp1d(invalid['rownum'], invalid['total'], fill_value="extrapolate")
    train['total'] = f(train['rownum'])
    del train['rownum']
