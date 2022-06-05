from control_limits.datasets.synthetic_data import load_data
from control_limits import ControlLimits, plot_control_limits


if __name__ == "__main__":
    # Load data benchmark dataset
    train_data, test_data, train_labels, test_labels = load_data()

    # Training of the control limits
    # Initialize the classifier
    clf = ControlLimits(precision_limits=0.75, length_limits=2, shape_limits=1)

    training_output = clf.train(train_data, train_labels)
    train_acc = clf.accuracy(train_data, train_labels, training_output)
    print(f'Training...\n'
          f'--> Accuracy: {train_acc}')

    # Testing of the control limits
    testing_output = clf.test(test_data, test_labels, training_output)
    test_acc = clf.accuracy(test_data, test_labels, testing_output)
    print(f'Testing...\n'
          f'--> Accuracy: {test_acc}')

    # Visualize the output
    plot_control_limits(train_data, test_data, train_labels, test_labels, training_output, testing_output)
