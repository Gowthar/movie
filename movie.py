import numpy as np
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#1
iris = ds.load_iris()

#print(iris.data)
#print(iris.target)
#print(iris.target_names)

#2
data_set = np.column_stack((iris.data, iris.target))
training_set, test_set = train_test_split(data_set, train_size=.70, test_size=.3)

#3
classifier = GaussianNB()
model = classifier.fit(training_set[:, :4], training_set[:, -1])

#4
targets_predicted = model.predict(test_set[:, :4])
#print(targets_predicted)

def get_accuracy(predicted,expected):
    count = 0.0
    total = len(expected)
    for i in range(total):
        if predicted[i] != expected[i]:
            count += 1
    return (total - count) / total

print(get_accuracy(targets_predicted, test_set[:, -1]))

#5
class HardCodedClassifer:
    def fit(self,data,target):
        return HardCodedModel()


class HardCodedModel:
    def predict(self,data_test):
        targets = np.zeros(len(data_test))
        for target in targets:
            target = 0
        return targets


classifier = HardCodedClassifer()
model = classifier.fit(training_set[:, :4], training_set[:, -1])
targets_predicted = model.predict(test_set[:, :4])


#print(get_accuracy(targets_predicted, test_set[:, -1]))

#ABOVE AND BEYOND
try:
    with open("flower_power.txt") as f:
        flower = f.read()
except FileNotFoundError:
    flower = None

print(flower)

