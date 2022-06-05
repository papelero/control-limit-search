# To do.

## Introduction
To do.

For installing the package, please read [Installation](#installation)

For using the package, please read [Instructions](#instructions)

For contributing, please read the [contribution guide](CONTRIBUTING.md).

## Installation 
> **_NOTE:_** You may skip these steps if you have already performed them.

1) Install [Poetry](https://python-poetry.org):
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```
2) Configure Poetry to not create a new virtual environment, instead reuse the Conda environment:
```
$ poetry config virtualenvs.create 0
```
3) Create a virtual environment using Python's builtin venv:
```
$ python -m venv .venv 
$ source .venv/bin/activate
```
or using Conda: 
```
conda create -n ENV_NAME python=X.Y
conda activate ENV_NAME
```
5) Install runtime and development dependencies:
```
$ poetry install 
```
6) Once the package is sucessfully installed, follow the instructions for the proper usage.

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

3. *Freezer Small Train* includes data from 20 households from Loughborough area over the period 2013-2014. 
For more information, see [https://timeseriesclassification.com/description.php?Dataset=FreezerSmallTrain](https://timeseriesclassification.com/description.php?Dataset=FreezerSmallTrain).

4. *SonyAIBORobotSurface1* includes data from a robot that has roll, pitch and yaw accelerometers.  
For more information, see [https://timeseriesclassification.com/description.php?Dataset=SonyAIBORobotSurface1](https://timeseriesclassification.com/description.php?Dataset=SonyAIBORobotSurface1). 

Use the following to access the chinatown dataset:
   ```python
   import datasets.chinatown as chinatown
   
   x_train, x_test, y_train, y_test = chinatown.load_data()
   ```

Use the following to access the gunpoint old versus young dataset use:
   ```python
   import datasets.gunpoint_oldversusyoung as gunpoint_oldversusyoung
   
   x_train, x_test, y_train, y_test = gunpoint_oldversusyoung.load_data()
   ```

Use the following to access the gunpoint old versus young dataset use:
   ```python
   import datasets.freezer_small_train as freezer_small_train
   
   x_train, x_test, y_train, y_test = freezer_small_train.load_data()
   ```

Use the following to access the gunpoint old versus young dataset use:
   ```python
   import datasets.sony_ai_robot as sony_ai_robot
   
   x_train, x_test, y_train, y_test = sony_ai_robot.load_data()
   ```

#### 2.2 - Own dataset

If you wish to use the package with your own dataset, make sure that the dataset has the following structure:
1. ```x_train``` and ```x_test``` must be two-dimensional numpy arrays of time-series.
2. ```y_train``` and ```y_test``` must be one-dimensional integer numpy arrays. The label of the first class 
must be 1, while the label of the second class must be 2.

### 3 - Running the package

#### 3.1 - Initialize and train the classifier

To initialize and train the classifier, use the following:

 ```python
 import control_limits_search
    
 cl = control_limits_search.ControlLimitsSearch(cl_precision=0.95, cl_len=10, cl_shape=1)
 train_pred = cl.train(x_train, y_train)
 ```
 
 The classifier has three parameters that one can change to improve/adjust the classification:
 1. ```cl_precision```, which represents the minimum desired precision that the classifier must attempt to preserve.
 2. ```cl_len```, which represents the minimum length of the control limits.
 3. ```cl_shape```, which represents whether the control limits are parallel (```cl_shape=1```) or not
parallel (```cl_shape=0```).

Use the following method to access the performance of the classifier:

```python
 train_precision, train_recall, train_f1_score  = cl.performance(x_train, y_train, train_pred)
 ```

#### 3.2 - Test the classifier

To evaluate the performance of the classifier on the test set, use the following:

 ```python
 test_pred = cl.test(x_test, y_test, train_pred)
 test_precision, test_recall, test_f1_score = cl.performance(x_test, y_test, test_pred)
 ```
 
 #### 3.3 - Visualizing the performance of the classifier
 
To visualize the performance of the classifier on the train and test set, we provide a function that plots the time 
series of the first class along with the false negative/false positive resulting from the classification task: 
```python
import plotting.plot as plot

plot.cl_plot(x_train, x_test, y_train, y_test, train_pred, test_pred, filename=filename)
```
The resulting plot is saved according to the file name provided from the user.

#### 3.4 - Example script

For further reference/demonstration on how to use the package, we provide an additional script in the example folder.

## Contributing
Improvements are always welcome, feel free to log a bug, write a suggestion or contribute code via merge request. 
All details are listed in our [contribution guide](CONTRIBUTING.md).

