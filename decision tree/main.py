import math
import numpy as np
import pandas as pd
import pprint
from collections import Counter
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


class DecisionTreeModel:
    def __init__(self, tree):
        self.tree = tree

    def show(self):
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.tree)

    def visit(self, row, node):
        if type(node) == dict:
            for key, subnode in node.items():
                if type(subnode) == dict and \
                type(key) == int and \
                row[key] in subnode:
                    return self.visit(row, subnode[row[key]])
                else:
                    return self.visit(row, subnode)
        else:
            return node

    def predict(self, data):
        tree = self.tree
        predictions = []
        for row in data:
            predictions.append(self.visit(row, tree))
        return np.array(predictions)


class DecisionTreeClassifier:
    def __init__(self):
        pass

    def calc_individual_entropy(self, a, total):
        # Cannot do log(0)
        if a == 0:
            return 0
        entropy = -(a / total) * math.log2(a / total)
        return entropy

    def calc_individual_entropies(self, targets):
        # Get unique target values
        unique_targets = list(set(targets))
        entropies = []
        for target in unique_targets:
            num_occurrences = (targets == target).sum()
            ent = self.calc_individual_entropy(num_occurrences, len(targets))
            entropies.append((target, ent))

        return entropies

    def calc_entropy(self, column, value, targets):
        # Get unique target values
        unique_targets = list(set(targets))

        # Find number of times value resulted in target
        counts = []
        for target in unique_targets:
            count = 0
            for i in range(0, len(targets)):
                if column[i] == value and targets[i] == target:
                    count += 1

            counts.append(count)

        # Find total entropy for value in column
        entropy = 0.0
        value_count = (column == value).sum()
        for count in counts:
            entropy += self.calc_individual_entropy(count, value_count)

        return entropy

    def calc_info_gain(self, data, targets, index):
        # Get total entropy
        num_targets = len(targets)
        unique_targets = list(set(targets))
        total_entropy = 0.0
        for target in unique_targets:
            count = (targets == target).sum()
            entropy = self.calc_individual_entropy(count, num_targets)
            total_entropy += (count / num_targets) * entropy

        # Get column to find information gain for feature
        column = data[:, index:index+1]
        column = np.concatenate(column, axis=0)
        column_size = len(column)

        # Get unique values within that column
        info_gain = total_entropy
        unique_values = list(set(column))
        for value in unique_values:
            # Get number of times value appears
            count = (column == value).sum()

            # Calculate entropy for column value
            entropy = self.calc_entropy(column, value, targets)

            # Info gain is the difference between column entropy
            # and total entropy
            info_gain -= (count / column_size) * entropy

        return info_gain

    def make_node(self, info_gains, data, targets, column_names):
        # If this is a leaf, figure out the correct value
        if len(info_gains) == 0:
            return {}

        # Get highest info gain
        ig = info_gains[0]
        info_gains = info_gains[1:]

        # Find index of matching feature
        index = list(column_names).index(ig[0])

        # Get entropies for feature
        column = data[:, index:index+1]
        column = np.concatenate(column, axis=0)
        ents = self.calc_individual_entropies(column)

        # Get greater entropy first
        ents = sorted(ents, key=lambda x: x[1])

        node = {ig[0]: {}}
        for ent in ents:
            if ents.index(ent) == 0 or len(info_gains) == 0:
                # Get associated results for value
                values = []
                for i in range(0, len(column)):
                    if column[i] == ent[0]:
                        values.append(targets[i])
                # Get best result
                mode = Counter(values).most_common()
                node[ig[0]][ent[0]] = mode[0][0]
            else:
                # Build tree
                node[ig[0]][ent[0]] = self.make_node(info_gains, data, targets, column_names)

        return node

    def make_tree(self, data, targets, column_names):
        # Find greatest information gain
        num_columns = len(data[0])
        info_gains = []
        for i in range(0, num_columns):
            column = column_names[i]
            ig = self.calc_info_gain(data, targets, i)
            info_gains.append((column, ig))

        # Sort by greatest information gain first
        info_gains = sorted(info_gains, key=lambda x: x[1], reverse=True)

        # Make tree
        tree = self.make_node(info_gains, data, targets, column_names)

        return tree

    def fit(self, data, targets, column_names):
        tree = self.make_tree(data, targets, column_names)
        return DecisionTreeModel(tree)


def get_data_movie_profit():
    headers = [
        "row",
        "type",
        "plot",
        "stars",
        "profit"
        ]
    df = pd.read_csv(
        "datasets/movie-profit.csv",
        header=None,
        names=headers,
        index_col=False
        )

    # Convert to numpy array
    array = df.values
    #print(df)
    #print(array)

    # Return the train and target data from the array
    #   The first slice returns the train data columns (all but the last column)
    #   The last slice returns the target data (the last column)
    return array[:,1:-1], array[:,-1], df.columns.values[1:-1]


def get_data_lenses():
    headers = [
        "linenum",
        "age",
        "prescription",
        "astig",
        "tear",
        "target"
        ]
    df = pd.read_csv(
        "datasets/lenses.data",
        header=None,
        delimiter="\s+",
        names=headers,
        index_col=False
        )

    # Get rid of unneeded lineum column
    df = df.drop("linenum", axis=1)

    # Convert to numpy array
    array = df.values
    #print(df)
    #print(array)

    # Return the train and target data from the array
    #   The first slice returns the train data columns (all but the last column)
    #   The last slice returns the target data (the last column)
    return array[:,:-1], array[:,-1], df.columns.values[:-1]


def main():
    # Load the datasets
    print("Available datasets:")
    print("1. Movie Profit")
    print("2. Lenses")

    print("")
    choice = input("Selection: ")

    if choice == "1":
        data, target, column_names = get_data_movie_profit()
    elif choice == "2":
        data, target, column_names = get_data_lenses()
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

    #print("data_train:", data_train)
    #print("data_test:", data_test)
    #print("targets_train:", targets_train)
    #print("targets_test:", targets_test)

    # Use classifier to create model, then predict test targets based
    # on the model's training
    classifier = DecisionTreeClassifier()
    model = classifier.fit(data_train, targets_train, column_names)

    print("model.predict()")
    targets_predicted = model.predict(data_test)

    print("model.show()")
    model.show()

    print("targets_predicted:", targets_predicted)
    print("targets_test:", targets_test)
    print("Accuracy:", get_accuracy(targets_predicted, targets_test))


if __name__ == "__main__":
    main()
