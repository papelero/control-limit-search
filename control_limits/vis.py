import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_limits(x_train, x_test, y_train, y_test, output_train, output_test):
    """Visualize the result of the Window Finder

    :param x_train: train data
    :type x_train: numpy array
    :param x_test: test data
    :type x_test: numpy array
    :param y_train: train labels
    :type y_train: numpy array
    :param y_test: test labels
    :type y_test: numpy array
    :param output_train: output training
    :type output_train: dict
    :param output_test: output testing
    :type output_test: dict
    """

    fig, axs = plt.subplots(1, 2, figsize=(5.512, 2.168))

    axs[0].plot(x_train[y_train == 1, :].T, color='darkgreen', lw=0.5, alpha=0.1, zorder=0)
    time_steps, boundaries = output_train['time_steps'], output_train['boundaries']
    data_fn, data_fp = output_train['fn'], output_train['fp']
    for fp in data_fp[0]:
        if fp.size != 0:
            axs[0].plot(fp, color='darkred', lw=0.5, label='FP')
    for fn in data_fn[0]:
        if fn.size != 0:
            axs[0].plot(fn, color='darkgreen', lw=0.5, ls='--', label='FN')
    for i in range(len(time_steps)):
        axs[0].plot(time_steps[i], boundaries[i][0], color='navy', ls='--', lw=0.8, zorder=2)
        axs[0].plot([time_steps[i][0], time_steps[i][0]], [boundaries[i][0][0], boundaries[i][1][0]],
                    color='navy', ls='--', lw=0.8, zorder=2)

        axs[0].plot(time_steps[i], boundaries[i][1], color='navy', ls='--', lw=0.8, zorder=2)
        axs[0].plot([time_steps[i][-1], time_steps[i][-1]], [boundaries[i][0][-1], boundaries[i][1][-1]],
                    color='navy', ls='--', lw=0.8, zorder=2)

        axs[0].fill_between(time_steps[i], boundaries[i][0], boundaries[i][1], facecolor='navy', alpha=0.25)
    axs[0].set_xlabel('$X$', fontsize=7)
    axs[0].set_ylabel('$Y$', fontsize=7)
    axs[0].set_title('Training', fontsize=7)
    xmin, xmax = axs[0].get_xaxis().get_view_interval()
    ymin, ymax = axs[0].get_yaxis().get_view_interval()
    axs[0].set_xlim([xmin, xmax])
    axs[0].set_xticks(np.round(np.linspace(xmin, xmax, 3)))
    axs[0].set_ylim([ymin, ymax])
    axs[0].set_yticks(np.round(np.linspace(ymin, ymax, 3)))
    axs[0].tick_params(axis='both', labelsize=7)
    handles, labels = axs[0].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if len(unique) != 0:
        axs[0].legend(*zip(*unique),
                      bbox_to_anchor=(0.275, 0.675),
                      fontsize=7,
                      edgecolor='white',
                      facecolor='white')

    axs[1].plot(x_test[y_test == 1, :].T, color='darkgreen', lw=0.5, alpha=0.1, zorder=0)
    time_steps, boundaries = output_test['time_steps'], output_test['boundaries']
    data_fn, data_fp = output_test['fn'], output_test['fp']
    for fp in data_fp[0]:
        if fp.size != 0:
            axs[1].plot(fp, color='darkred', lw=0.5, label='FP')
    for fn in data_fn[0]:
        if fn.size != 0:
            axs[1].plot(fn, color='darkgreen', lw=0.5, ls='--', label='FN')
    for i in range(len(time_steps)):
        axs[1].plot(time_steps[i], boundaries[i][0], color='navy', ls='--', lw=0.8, zorder=2)
        axs[1].plot([time_steps[i][0], time_steps[i][0]], [boundaries[i][0][0], boundaries[i][1][0]],
                    color='navy', ls='--', lw=0.8, zorder=2)

        axs[1].plot(time_steps[i], boundaries[i][1], color='navy', ls='--', lw=0.8, zorder=2)
        axs[1].plot([time_steps[i][-1], time_steps[i][-1]], [boundaries[i][0][-1], boundaries[i][1][-1]],
                    color='navy', ls='--', lw=0.8, zorder=2)

        axs[1].fill_between(time_steps[i], boundaries[i][0], boundaries[i][1], facecolor='navy', alpha=0.25)
    axs[1].set_xlabel('$X$', fontsize=7)
    axs[1].set_title('Testing', fontsize=7)
    xmin, xmax = axs[1].get_xaxis().get_view_interval()
    ymin, ymax = axs[1].get_yaxis().get_view_interval()
    axs[1].set_xlim([xmin, xmax])
    axs[1].set_xticks(np.round(np.linspace(xmin, xmax, 3)))
    axs[1].set_ylim([ymin, ymax])
    axs[1].set_yticks(np.round(np.linspace(ymin, ymax, 3)))
    axs[1].set_yticklabels([])
    axs[1].tick_params(axis='both', labelsize=7)
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
