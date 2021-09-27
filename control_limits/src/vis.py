import matplotlib.pyplot as plt
import seaborn as sns


def plot_control_limits(train_data, test_data, train_labels, test_labels, training_output, testing_output):
    """Show the control limits

    :param train_data: train data
    :param test_data: test data
    :param train_labels: train labels
    :param test_labels: test labels
    :param training_output: output training
    :param testing_output: output testing
    """

    fig, axs = plt.subplots(1, 2, figsize=(5.512, 2.168))

    axs[0].plot(train_data[train_labels == 1, :].T, color='darkgreen', lw=0.5, alpha=0.1, zorder=0)
    time_steps, control_limits = training_output["time_steps"], training_output["control_limits"]
    data_false_negative, data_false_positive = training_output["false_negative"], training_output["false_positive"]
    for false_positive in data_false_positive[0]:
        if false_positive.size != 0:
            axs[0].plot(false_positive, color='darkred', lw=0.5, label='FP')
    for false_negative in data_false_negative[0]:
        if false_negative.size != 0:
            axs[0].plot(false_negative, color='darkgreen', lw=0.5, ls='--', label='FN')
    for i in range(len(time_steps)):
        axs[0].plot(time_steps[i], control_limits[i][0], color='navy', ls='--', lw=0.8, zorder=2)
        axs[0].plot([time_steps[i][0], time_steps[i][0]], [control_limits[i][0][0], control_limits[i][1][0]],
                    color='navy', ls='--', lw=0.8, zorder=2)

        axs[0].plot(time_steps[i], control_limits[i][1], color='navy', ls='--', lw=0.8, zorder=2)
        axs[0].plot([time_steps[i][-1], time_steps[i][-1]], [control_limits[i][0][-1], control_limits[i][1][-1]],
                    color='navy', ls='--', lw=0.8, zorder=2)

        axs[0].fill_between(time_steps[i], control_limits[i][0], control_limits[i][1], facecolor='navy', alpha=0.25)
    axs[0].set_xlabel('$X$', fontsize=7)
    axs[0].set_ylabel('$Y$', fontsize=7)
    axs[0].set_title('Training', fontsize=7)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    handles, labels = axs[0].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if len(unique) != 0:
        axs[0].legend(*zip(*unique),
                      bbox_to_anchor=(0.275, 0.675),
                      fontsize=7,
                      edgecolor='white',
                      facecolor='white')

    axs[1].plot(test_data[test_labels == 1, :].T, color='darkgreen', lw=0.5, alpha=0.1, zorder=0)
    time_steps, control_limits = testing_output["time_steps"], testing_output["control_limits"]
    data_false_negative, data_false_positive = testing_output["false_negative"], testing_output["false_positive"]
    for false_positive in data_false_positive[0]:
        if false_positive.size != 0:
            axs[1].plot(false_positive, color='darkred', lw=0.5, label='FP')
    for false_negative in data_false_negative[0]:
        if false_negative.size != 0:
            axs[1].plot(false_negative, color='darkgreen', lw=0.5, ls='--', label='FN')
    for i in range(len(time_steps)):
        axs[1].plot(time_steps[i], control_limits[i][0], color='navy', ls='--', lw=0.8, zorder=2)
        axs[1].plot([time_steps[i][0], time_steps[i][0]], [control_limits[i][0][0], control_limits[i][1][0]],
                    color='navy', ls='--', lw=0.8, zorder=2)

        axs[1].plot(time_steps[i], control_limits[i][1], color='navy', ls='--', lw=0.8, zorder=2)
        axs[1].plot([time_steps[i][-1], time_steps[i][-1]], [control_limits[i][0][-1], control_limits[i][1][-1]],
                    color='navy', ls='--', lw=0.8, zorder=2)

        axs[1].fill_between(time_steps[i], control_limits[i][0], control_limits[i][1], facecolor='navy', alpha=0.25)
    axs[1].set_xlabel('$X$', fontsize=7)
    axs[1].set_title('Testing', fontsize=7)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    handles, labels = axs[1].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    axs[1].legend(*zip(*unique),
                  bbox_to_anchor=(0.275, 0.675),
                  fontsize=7,
                  edgecolor='white',
                  facecolor='white')
    sns.despine()
    fig.tight_layout()
    plt.show()
