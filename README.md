# Boston Housing Price Prediction

This project implements a feed-forward neural network model to predict house prices based on various features of houses in the Boston area. The model is built using TensorFlow and trained on the Boston Housing dataset.

## Dataset

The **Boston Housing dataset** contains information about various features of homes in Boston and their corresponding prices. Each row in the dataset represents a unique home, and each column provides information about a specific feature, such as crime rate, average number of rooms, accessibility to highways, etc.

### Features

The dataset includes the following features:
- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitrogen oxide concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built before 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: Proportion of Black population by town
- **LSTAT**: Percentage of lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000s (target variable)

## Project Structure

The project includes the following files:

- **boston_housing_model.py**: Contains the `BostonHousingModel` class, which builds and trains the neural network, and the `load_boston_data` function to load and preprocess the dataset.
- **Boston_Housing_Prediction.ipynb**: A Jupyter notebook demonstrating the steps to load data, train the model, and evaluate performance.
- **README.md**: This file, containing an overview and instructions for using the project.

## Model Architecture

The neural network model is a simple feed-forward architecture with the following layers:

- **Input Layer**: Accepts 13 features (one for each dataset feature).
- **Hidden Layers**:
  - Dense layer with 64 neurons and ReLU activation
  - Dense layer with 32 neurons and ReLU activation
- **Output Layer**: Single neuron for regression (predicting house price)

The model is compiled with **mean squared error (MSE)** as the loss function, which is suitable for regression tasks, and **mean absolute error (MAE)** as an evaluation metric.

## Setup and Usage

### Requirements

Ensure that Python and the following libraries are installed:

```bash
pip install tensorflow pandas scikit-learn
