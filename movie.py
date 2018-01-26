import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

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
    def __init__(self, data, targets):
        self.k = 3
        self.data = data
        self.targets = targets

    def predict(self, test_data):
        # Predict each element in dataset
        targets = []
        for element in test_data:
            predicted_target = self.predict_one(element)
            targets.append(predicted_target)

        return targets

    def predict_one(self, test_element):
        test_element_size = len(test_element)
        training_data_size = len(self.data)
        distance_list = []
        for i in range(training_data_size):
            distance = 0.0
            for j in range(test_element_size):
                distance += (test_element[j] - self.data[i][j]) ** 2
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


class KNNClassifier:
    def __init__(self):
        pass

    def fit(self, data, targets):
        return KNNModel(data, targets)



#data = pd.read_csv('adult.data.csv', names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','#'])

#def get_data_car(filename):
 #   data = pd.read_csv(filename)
  #  print(data)


#def main():
############Read in label columns#################
data = pd.read_csv('car.data', names = ['buying','maint','doors','persons','lug_boot','safety'], index_col=False)
######################Replacing objects with numeric scales#############################
data['buying'].replace(to_replace=['vhigh','high','med','low'], value =[4,3,2,1], inplace=True)
data['maint' ].replace(to_replace=['vhigh','high','med','low'], value =[4,3,2,1], inplace=True)
data['doors'].replace(to_replace=['2','3','4','5more'], value =[1,2,3,4], inplace=True)
data['persons'].replace(to_replace=['2','4','more'], value =[1,2,3], inplace=True)
data['lug_boot'].replace(to_replace=['small','med','big'], value =[1,2,3], inplace=True)
data['safety'].replace(to_replace=['low','med','high'], value =[1,2,3], inplace=True)
#print(data.head(5))
######################Converting Pandas to nparray############################################
data = data.values
print(data)

    # Load the datasets
'''
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    data_train, data_test, targets_train, targets_test = train_test_split(
        data,
        target,
        train_size=0.7,
        test_size=0.3,
        shuffle=True
        )

    classifier = KNNClassifier()
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    print(get_accuracy(targets_predicted, targets_test))
'''

#if __name__ == "__main__":
    #main()

#Worked with Josh Backstein