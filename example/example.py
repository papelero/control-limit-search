# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os

import datasets.chinatown as chinatown
import control_limits_search
import plotting.plot as plot

if __name__ == "__main__":

    # Load the data
    x_train, x_test, y_train, y_test = chinatown.load_data()

    # Initialize and search for the control limits on the train data
    cl = control_limits_search.ControlLimitsSearch(cl_precision=1.0, cl_len=2, cl_shape=1)
    train_pred = cl.train(x_train, y_train)

    # Test the control limits
    test_pred = cl.test(x_test, y_test, train_pred)

    # Plot the outcome
    filename = os.path.join(os.getcwd(), "example.pdf")
    plot.cl_plot(x_train, x_test, y_train, y_test, train_pred, test_pred, filename=filename)
