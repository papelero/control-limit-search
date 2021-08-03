# Control limits for statistical process control 

## Introduction

This GitHub project is a stand-alone package that uses statistical techniques to search for optimal control limits that linearly separate two classes of the time series data. It has been successfully employed in the automotive industry to identify optimal control limits for fault detection in tightening process data.
Although the algorithm was inspired by applications in the automotive industry, it can be applied for the binary classification of generic multivariate time series problems. 

For the installation of the package, please read [Installation](#installation).

For using the package, please read [Instructions](#instructions).

## Installation

Install the package using the following instructions:
1. Clone the repository to download the source distribution ([https://github.com/papelero/control-limit-search.git](https://github.com/papelero/control-limit-search.git)).
2. Access the root directory where ```setup.py``` is located.
3. Run the following commands in the terminal (Linux) or command prompt window (Windows):
  ```python
  python setup.py install
  pip install -r requirements.txt
  ```

## Instructions

### 1 - Installation

Install the package following the instructions of [Installation](#installation).

### 2 - Datasets

#### 2.1 - Available datasets

The package has two multivariate time series datasets available:
1. *Chinatown* derived from the automated pedestrian counting system. The two classes are the pedestrian count on weekdays as opposed to weekend. For more information, see [https://timeseriesclassification.com/description.php?Dataset=Chinatown](https://timeseriesclassification.com/description.php?Dataset=Chinatown). 

2. *Gunpoint Old versus Young* originates from the surveillance domain and is a remake of the popular *Gunpoint* dataset. The two classes differ according to whether the actor is holding a gun or not during the measurement. For more information, see [https://timeseriesclassification.com/description.php?Dataset=GunPointOldVersusYoung](https://timeseriesclassification.com/description.php?Dataset=GunPointOldVersusYoung). 

To access the chinatown dataset, use the following:
   ```python
   from control_limits.datasets.chinatown import data_loader
   
   x_train, x_test, y_train, y_test = data_loader()
   ```
   
To access the gunpoint old versus young dataset, use the following:
   ```python
   from control_limits.datasets.gunpoint_oldversusyoung import data_loader
   
   x_train, x_test, y_train, y_test = data_loader()
   ```

#### 2.2 - Own dataset

If you wish to apply the package to your own dataset, make sure it has the following structure:
1. ```x_train``` and ```x_test``` must be two-dimensional numpy arrays.
2. ```y_train``` and ```y_test``` must be one-dimensional integer numpy arrays, where the label of the first class is 1 and the label of the second class is 2.

If the following structure is not observed, the package will raise an error.

### 3 - Running the package

#### 3.1 - Initialize and train the classifier

To initialize and train the classifier, use the following:

 ```python
 from control_limits import ControlLimits
    
 clf = ControlLimits(x_train, y_train)
 output_train = clf.fit()
 ```
 
 The classifier has three optional parameters that one can change to improve/adjust the classification:
 1. ```precision_limits``` representing the minimum desired precision that the classifier must attempt to preserve (Default: 0.95).
 2. ```length_limits``` representing the minimum length of the control limits (Default: 8).
 3. ```shape_limits``` representing whether the control limits are parallel (1) or not (0) (Default: 1).
 
 To change the parameters, initialize the classifier as follows: 
 
 ```python
 from control_limits import ControlLimits
 
 clf = ControlLimits(x_train, y_train, precision_limits=0.75, length_limits=2, shape_limits=0)
 ```
 
To access the perfomance of the classifier on the train set, use the followinng:

```python
 train_acc = clf.accuracy(x_train, y_train, output_train)
 ```

#### 3.2 - Evaluate the classifier

To evaluate the performance of the classifier on the test set, do the following:

 ```python
 from control_limits import ControlLimits
    
 clf = ControlLimits(x_train, y_train)
 output_clf = clf.fit()
 
 output_test = clf.evaluate(test_x, test_y, output_train)
 test_acc = clf.accuracy(x_test, y_test, output_test)
 ```
 
 #### 3.3 - Visualizing the performance of the classifier
 
To visualize the performance of the classifier on the train and test set, we provide a method that plots the time series of the first class along with resulting false negative/false positive.
To use the method, do the following: 
```python
from control_limits.utils import plot_limits

plot_limits(x_train, x_test, y_train, y_test, output_train, output_test)
```

#### 3.4 - Example script

For further reference/demonstration on how to use the package, we provide an example script in ```examples/example.py``` that runs on a dataset artificially constructed.










