import matplotlib.pyplot as plt

def plot_scatter(x_train, x_test, y_train, y_test, train_predict, test_predict):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel('x_train')
    plt.ylabel('y_train')
    for i in range(len(x_train)):
        plt.scatter(x_train[i], y_train[i])
    plt.plot(x_train, train_predict, color='red', linewidth=2)

    plt.subplot(1, 2, 2)
    plt.xlabel('x_test')
    plt.ylabel('y_test')
    for i in range(len(x_test)):
        plt.scatter(x_test[i], y_test[i])
    plt.plot(x_test, test_predict, color='red', linewidth=2)

    plt.show()

def plot_loss_and_accuracy(self, losses, accs):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    axes[0].plot(losses)
    axes[1].plot(accs)
    axes[1].set_xlabel("Epoch", fontsize=15)
    axes[0].set_ylabel("Loss", fontsize=15)
    axes[1].set_ylabel("Accuracy", fontsize=15)
    axes[0].tick_params(labelsize=10)
    axes[1].tick_params(labelsize=10)

    fig.tight_layout()
    plt.show()