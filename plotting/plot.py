import matplotlib.pyplot as plt


def cl_plot(x_train, x_test, y_train, y_test, train_pred, test_pred, *, filename=None):
    """Plot the control limits

    :param x_train: train data
    :param x_test: test data
    :param y_train: train labels
    :param y_test: test labels
    :param train_pred: output training
    :param test_pred: output testing
    :param filename: path and name of file to store output
    """

    fig, axs = plt.subplots(1, 2, figsize=(5.512, 2.168))

    # Plot the normal data
    axs[0].plot(x_train[y_train == 1, :].T, color="darkgreen", lw=0.5, alpha=0.1, zorder=0)

    # Plot the false positive
    for false_positive in train_pred["fp"][0]:
        if false_positive.size != 0:
            axs[0].plot(false_positive, color="darkred", lw=0.5, label="fp")

    # Plot the false negative
    for false_negative in train_pred["fn"][0]:
        if false_negative.size != 0:
            axs[0].plot(false_negative, color="darkgreen", lw=0.5, ls="--", label="fn")

    # Plot the control limits
    for i in range(len(train_pred["ts"])):
        axs[0].plot(train_pred["ts"][i], train_pred["cl"][i][0], color="navy", ls="--", lw=1.0, zorder=2)
        axs[0].plot([train_pred["ts"][i][0], train_pred["ts"][i][0]],
                    [train_pred["cl"][i][0][0], train_pred["cl"][i][1][0]],
                    color="navy", ls="--", lw=1.0, zorder=2)
        axs[0].plot(train_pred["ts"][i], train_pred["cl"][i][1], color="navy", ls="--", lw=1.0, zorder=2)
        axs[0].plot([train_pred["ts"][i][-1], train_pred["ts"][i][-1]],
                    [train_pred["cl"][i][0][-1], train_pred["cl"][i][1][-1]],
                    color="navy", ls="--", lw=1.0, zorder=2)
        axs[0].fill_between(train_pred["ts"][i], train_pred["cl"][i][0], train_pred["cl"][i][1],
                            facecolor="navy", alpha=0.15)
    axs[0].set_xlabel("$X$", fontsize=7)
    axs[0].set_ylabel("$Y$", fontsize=7)
    axs[0].set_title("Training prediction", fontsize=7)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    handles, labels = axs[0].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if len(unique) != 0:
        axs[0].legend(*zip(*unique),
                      loc="best",
                      fontsize=7,
                      edgecolor="white",
                      facecolor="white")

    # Plot the normal data
    axs[1].plot(x_test[y_test == 1, :].T, color="darkgreen", lw=0.5, alpha=0.1, zorder=0)

    # Plot the false positive
    for false_positive in test_pred["fp"][0]:
        if false_positive.size != 0:
            axs[1].plot(false_positive, color="darkred", lw=1.0, label="fp")

    # Plot the false negative
    for false_negative in test_pred["fn"][0]:
        if false_negative.size != 0:
            axs[1].plot(false_negative, color="darkgreen", lw=1.0, ls="--", label="fn")

    # Plot the control limits
    for i in range(len(test_pred["ts"])):
        axs[1].plot(test_pred["ts"][i], test_pred["cl"][i][0], color="navy", ls="--", lw=1.0, zorder=2)
        axs[1].plot([test_pred["ts"][i][0], test_pred["ts"][i][0]],
                    [test_pred["cl"][i][0][0], test_pred["cl"][i][1][0]],
                    color="navy", ls="--", lw=1.0, zorder=2)
        axs[1].plot(test_pred["ts"][i], test_pred["cl"][i][1], color="navy", ls="--", lw=1.0, zorder=2)
        axs[1].plot([test_pred["ts"][i][-1], test_pred["ts"][i][-1]],
                    [test_pred["cl"][i][0][-1], test_pred["cl"][i][1][-1]],
                    color="navy", ls="--", lw=1.0, zorder=2)
        axs[1].fill_between(test_pred["ts"][i], test_pred["cl"][i][0], test_pred["cl"][i][1],
                            facecolor="navy", alpha=0.15)
    axs[1].set_xlabel("$X$", fontsize=7)
    axs[1].set_title("Testing prediction", fontsize=7)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    handles, labels = axs[1].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    axs[1].legend(*zip(*unique),
                  loc="best",
                  fontsize=7,
                  edgecolor="white",
                  facecolor="white")
    fig.tight_layout()
    if filename:
        plt.savefig(filename, dpi=1000)
    plt.show()
