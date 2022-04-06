from matplotlib import pyplot as plt


def test():
    print(69)


def plot_the_loss_curve(epochs, error):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, error, label="Loss")
    plt.legend()
    plt.ylim([error.min()*0.94, error.max()* 1.05])
    plt.show()
