# Automated search of process control limits for fault detection in time series data 

## Introduction

This GitHub repo is a stand-alone package that uses statistical/information theory techniques to automate the search 
of optimal control limits for fault detection in time series data. 
It has been successfully employed in the automotive industry to perform fault detection in tightening process data.

Although the algorithm was inspired by applications in the automotive industry, it can be applied for the traditional 
time series classification tasks. 

For the installation of the package, please read [Installation](#installation).

For using the package, please read [Instructions](#instructions).

## Installation

Install the package using the following instructions:
1. Clone the repository to download the source distribution ([https://github.com/tomaselli96/control_limits](https://github.com/tomaselli96/control_limits)).
2. Access the root directory where ```setup.py``` is located.
3. Run the following commands in the terminal (Linux/Mac) or command prompt window (Windows):
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
1. *Chinatown* derived from the automated pedestrian counting system. For more information, 
see [https://timeseriesclassification.com/description.php?Dataset=Chinatown](https://timeseriesclassification.com/description.php?Dataset=Chinatown). 

2. *Gunpoint Old versus Young* originates from the surveillance domain and is a remake of the popular *Gunpoint* dataset. 
For more information, see [https://timeseriesclassification.com/description.php?Dataset=GunPointOldVersusYoung](https://timeseriesclassification.com/description.php?Dataset=GunPointOldVersusYoung). 

3. *Synthetic data* is a dataset generated artificially for demonstration and testing purposes. Nevertheless, we
added it to the list of available datasets and it can be utilized as the previous two. 

Use the following to access the chinatown dataset:
   ```python
   from control_limits.datasets.chinatown import load_data
   
   train_data, test_data, train_labels, test_labels = load_data()
   ```

Use the following to access the gunpoint old versus young dataset use:
   ```python
   from control_limits.datasets.gunpoint_oldversusyoung import load_data
   
   train_data, test_data, train_labels, test_labels = load_data()
   ```

Use the following to access the synthetic dataset:
   ```python
   from control_limits.datasets.synthetic_data import load_data
   
   train_data, test_data, train_labels, test_labels = load_data()
   ```


#### 2.2 - Own dataset

If you wish to use the package with your own dataset, make sure that the dataset has the following structure:
1. ```train_data``` and ```test_data``` must be two-dimensional numpy arrays.
2. ```train_labels``` and ```test_labels``` must be one-dimensional integer numpy arrays. The label of the first class 
must be 1, while the label of the second class must be 2.

### 3 - Running the package

#### 3.1 - Initialize and train the classifier

To initialize and train the classifier, use the following:

 ```python
 from control_limits import ControlLimits
    
 clf = ControlLimits(precision_limits=0.95, length_limits=10, shape_limits=1)
 training_output = clf.train(train_data, train_labels)
 ```
 
 The classifier has three parameters that one can change to improve/adjust the classification:
 1. ```precision_limits```, which represents the minimum desired precision that the classifier must attempt to preserve.
 2. ```length_limits```, which represents the minimum length of the control limits.
 3. ```shape_limits```, which represents whether the control limits are parallel (```shape_limits=1```) or not
parallel (```shape_limits=0```).

Use the following method to access the perfomance of the classifier:

```python
 train_acc = clf.accuracy(train_data, train_labels, training_output)
 ```

#### 3.2 - Test the classifier

To evaluate the performance of the classifier on the test set, use the following:

 ```python
 testing_output = clf.test(test_data, test_labels, training_output)
 test_acc = clf.accuracy(test_data, test_labels, testing_output)
 ```
 
 #### 3.3 - Visualizing the performance of the classifier
 
To visualize the performance of the classifier on the train and test set, we provide a function that plots the time 
series of the first class along with the false negative/false positive resulting from the classification task: 
```python
from control_limits import plot_control_limits

plot_limits(train_data, test_data, train_labels, test_labels, training_output, testing_output)
```

#### 3.4 - Example script

For further reference/demonstration on how to use the package, we provide an additional script in the example folder.
