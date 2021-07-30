import os
import sys

path = os.getcwd().split('\\')[:-1]
sys.path.append('\\'.join(path))

from examples.data_example import ExampleSeries
from control_limits.src import ControlLimits
from control_limits.vis import plot_limits


def main():
    """Executable function for running window finder."""

    x_train, x_test, y_train, y_test = ExampleSeries().generate(return_X_y=True)

    import matplotlib.pyplot as plt
    plt.plot(x_train[y_train == 1, :].T, color='darkgreen')
    plt.plot(x_train[y_train == 2, :].T, color='darkred')
    plt.show()

    clf = ControlLimits(x_train, y_train)
    output_train = clf.fit()
    train_acc = clf.accuracy(x_train, y_train, output_train)
    print(f'Training...\n'
          f'--> Accuracy: {train_acc}')

    output_test = clf.evaluate(x_test, y_test, output_train)
    test_acc = clf.accuracy(x_test, y_test, output_test)
    print(f'Testing...\n'
          f'--> Accuracy: {test_acc}')

    plot_limits(x_train, x_test, y_train, y_test, output_train, output_test)


if __name__ == '__main__':
    main()
