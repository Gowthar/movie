import math
import random
import pandas as pd
#import sklearn as sk
import nueral_network
import neuron_layer
import neuron
import csv
import urllib.request


BIAS = -1


def get_data_pima_indians_diabetes():
    headers = ["num_pregnant", "plasma_glucose_con", "diastolic_bp", "tri_thickness", "2hr_serum_insulin", "bmi", "diabetes_pedigree_function", "age", "class"]
    df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=headers, index_col=False)

    # Replace the null values with the mode of each column
    # Returns the mode of each column
    modes = df.mode().values[0]

    df['plasma_glucose_con'].replace(to_replace=[0], value=[modes[1]], inplace=True)
    df['diastolic_bp'].replace(to_replace=[0], value=[modes[2]], inplace=True)
    df['tri_thickness'].replace(to_replace=[0], value=[modes[3]], inplace=True)
    df['2hr_serum_insulin'].replace(to_replace=[0], value=[modes[4]], inplace=True)
    df['bmi'].replace(to_replace=[0], value=[modes[5]], inplace=True)
    df['diabetes_pedigree_function'].replace(to_replace=[0], value=[modes[6]], inplace=True)
    df['age'].replace(to_replace=[0], value=[modes[7]], inplace=True)

    # Convert the dataframe to a numpy array
    array = df.values

    # Return the train and target data from the array
    #   The first slice returns the train data columns (all but the last column)
    #   The last slice returns the target data (the last column)
    return array[:, :-1], array[:, -1]


def get_data_iris():
    headers = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = pd.read_csv("iris.csv", header=None, names=headers, index_col=False)

    # Replace the null values with the mode of each column
    # Returns the mode of each column
    modes = df.mode().values[0]

    df['sepal_length'].replace(to_replace=[0], value=[modes[1]], inplace=True)
    df['sepal_width'].replace(to_replace=[0], value=[modes[2]], inplace=True)
    df['petal_length'].replace(to_replace=[0], value=[modes[3]], inplace=True)
    df['petal_width'].replace(to_replace=[0], value=[modes[4]], inplace=True)

    # Convert the dataframe to a numpy array
    array = df.values

    # Return the train and target data from the array
    #   The first slice returns the train data columns (all but the last column)
    #   The last slice returns the target data (the last column)
    return array[:, :-1], array[:, -1]


"""
To view the structure of the Neural Network, type
print network_name
"""
