import numpy
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Get accuracy between the predicted targets and the expected targets
def get_accuracy(predicted, expected):
    count = 0.0
    total = len(expected)
    for i in range(total):
        if predicted[i] != expected[i]:
            count += 1

    return (total - count) / total


class HardCodedModel:
    def __init__(self):
        pass

    def predict(self, data):
        targets = numpy.zeros(len(data))
        return targets


class HardCodedClassifier:
    def __init__(self):
        pass

    def fit(self, data, targets):
        return HardCodedModel()


class KNNModel:
    def __init__(self, data, targets, k):
        self.k = k
        self.data = data
        self.targets = targets

        # Find max and min of data for normalization
        self.data_max = []
        self.data_min = []
        data_element_size = len(data[0])
        for i in range(data_element_size):
            data_max = max(self.data, key=lambda x: x[i])
            data_min = min(self.data, key=lambda x: x[i])
            self.data_max.append(data_max[i])
            self.data_min.append(data_min[i])

    def predict(self, test_data):
        # Predict each element in dataset
        targets = []
        for element in test_data:
            predicted_target = self.predict_one(element)
            targets.append(predicted_target)

        return targets

    def predict_one(self, test_element):
        """
        Find k nearest neighbors to predict the target
        """
        # Normalize test element to find proper distance from neighbors
        test_element_normalized = self.normalize_element(test_element)

        # Find distance between the test element and training data
        test_element_size = len(test_element)
        training_data_size = len(self.data)
        distance_list = []
        for i in range(training_data_size):
            # Normalize training data element to find proper distance from test element
            data_element_normalized = self.normalize_element(self.data[i])

            # Get distance between test element and training data element
            distance = 0.0
            for j in range(test_element_size):
                # (b - a)^2
                distance += (test_element_normalized[j] - data_element_normalized[j]) ** 2

            distance_list.append((distance, self.targets[i]))

        # Sort distance list
        # Source: https://stackoverflow.com/questions/10695139/sort-a-list-of-tuples-by-2nd-item-integer-value
        sorted_distance_list = sorted(distance_list, key=lambda x: x[0])

        # Get k nearest neighbors
        nearest_neighbors = sorted_distance_list[:self.k]

        # Find most common neighbor type
        types_of_nearest_neighbors = []
        for neighbor in nearest_neighbors:
            types_of_nearest_neighbors.append(neighbor[1])

        # Source: https://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
        predicted_type = max(types_of_nearest_neighbors, key=types_of_nearest_neighbors.count)
        return predicted_type

    def normalize_element(self, test_element):
        """
        Normalize the data in an element to fit within the range 0..1
        """
        test_element_size = len(test_element)
        normalized_element = []
        for i in range(test_element_size):
            normalized_element.append(
                # Source: https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
                (test_element[i] - self.data_min[i]) / (self.data_max[i] - self.data_min[i])
            )

        return normalized_element


class KNNClassifier:
    def __init__(self):
        pass

    def fit(self, data, targets, k):
        return KNNModel(data, targets, k)


def get_iris_dataset():
    iris = datasets.load_iris()

    return iris.data, iris.target


def get_data_uci_car_evaluation():
    headers = [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "class"
        ]
    df = pd.read_csv(
        "car-evaluation.data",
        header=None,
        names=headers,
        index_col=False
        )

    # Replace each column one by one
    df['buying'].replace(to_replace=['vhigh', 'high', 'med', 'low'], value=[4.0, 3.0, 2.0, 1.0], inplace=True)
    df['maint'].replace(to_replace=['vhigh', 'high', 'med', 'low'], value=[4.0, 3.0, 2.0, 1.0], inplace=True)
    df['doors'].replace(to_replace=['2', '3', '4', '5more'], value=[1.0, 2.0, 3.0, 4.0], inplace=True)
    df['persons'].replace(to_replace=['2', '4', 'more'], value=[1.0, 2.0, 3.0], inplace=True)
    df['lug_boot'].replace(to_replace=['small', 'med', 'big'], value=[1.0, 2.0, 3.0], inplace=True)
    df['safety'].replace(to_replace=['low', 'med', 'high'], value=[1.0, 2.0, 3.0], inplace=True)
    df['class'].replace(to_replace=['unacc', 'acc', 'good', 'vgood'], value=[1.0, 2.0, 3.0, 4.0], inplace=True)

    # Convert to numpy array
    array = df.values

    # Return the train and target data from the array
    #   The first slice returns the train data columns (all but the last column)
    #   The last slice returns the target data (the last column)
    return array[:,:-1], array[:,-1]


def get_data_pima_indians_diabetes():
    headers = [
        "num_pregnant",
        "plasma_glucose_con",
        "diastolic_bp",
        "tri_thickness",
        "2hr_serum_insulin",
        "bmi",
        "diabetes_pedigree_function",
        "age",
        "class"
        ]
    df = pd.read_csv(
        "pima-indians-diabetes.data",
        header=None,
        names=headers,
        index_col=False
        )

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


def get_data_automobile_mpg():
    headers = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name"
        ]
    df = pd.read_csv(
        "auto-mpg.data",
        header=None,
        names=headers,
        delim_whitespace=True,
        index_col=False
        )

    # Returns the mode of each column
    modes = df.mode().values[0]

    # Replace missing value in the horsepower column with the mode of the column
    df['horsepower'].replace(to_replace=['?'], value=[float(modes[3])], inplace=True)
    df = df.convert_objects(convert_numeric=True)

    # Need to move mpg to be the last column and drop the car_names
    # column, as it does not provide valuable information
    column_titles = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "mpg"
        ]
    df = df.reindex(columns=column_titles)

    # Convert the dataframe to a numpy array
    array = df.values

    # Return the train and target data from the array
    #   The first slice returns the train data columns (all but the last column)
    #   The last slice returns the target data (the last column)
    return array[:, :-1], array[:, -1]


def main():
    # Load the datasets
    print("Available datasets:")
    print("1. UCI: Car Evaluation")
    print("2. Pima Indians Diabetes")
    print("3. Automobile MPG")
    print("")
    choice = input("Selection: ")

    if choice == "1":
        classifier_type = "knn"
        data, target = get_data_uci_car_evaluation()
        k = 15
    elif choice == "2":
        classifier_type = "knn"
        data, target = get_data_pima_indians_diabetes()
        k = 25
    elif choice == "3":
        classifier_type = "linear_regression"
        data, target = get_data_automobile_mpg()
    else:
        print("Invalid input.")
        print("Exiting...")
        exit()

    data_train, data_test, targets_train, targets_test = train_test_split(
        data,
        target,
        train_size=0.7,
        test_size=0.3,
        shuffle=True
        )

    if classifier_type == "knn":
        classifier = KNNClassifier()
        model = classifier.fit(data_train, targets_train, k)
    elif classifier_type == "linear_regression":
        classifier = linear_model.LinearRegression()
        model = classifier.fit(data_train, targets_train)
    else:
        classifier = HardCodedClassifier()
        model = classifier.fit(data_train, targets_train)

    targets_predicted = model.predict(data_test)
    print(get_accuracy(targets_predicted, targets_test))

main()