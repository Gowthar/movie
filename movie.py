import numpy as np
from tkinter import *
from tkinter import ttk
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

#print(get_accuracy(targets_predicted, test_set[:, -1]))

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

accuracy = get_accuracy(targets_predicted, test_set[:, -1])


#ABOVE AND BEYOND

#try:
#    with open("flower_power.txt") as f:
#        flower = f.read()
#except FileNotFoundError:
#    flower = None

### READING LINES FROM TEXT files
path = "flower_power.txt"
lines = [line for line in open(path)]
print(lines[0])
z = lines[0]

#### OUTPUTING GUI

def calculate(*args):
    try:
        outy.set(accuracy)
    except ValueError:
        pass


root = Tk()
root.title("MACHINE LEARNING BRO")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

iny = StringVar()
outy = StringVar()

iny_entry = ttk.Entry(mainframe, width=7, textvariable=iny)
iny_entry.grid(column=2, row=1, sticky=(W, E))

ttk.Label(mainframe, textvariable=outy).grid(column=2, row=2, sticky=(W, E))
ttk.Button(mainframe, text="Calculate", command=calculate).grid(column=3, row=3, sticky=W)

ttk.Label(mainframe, text="Training Size (0.0 - 1.0)").grid(column=3, row=1, sticky=W)
ttk.Label(mainframe, text="Your Accuracy is ").grid(column=1, row=2, sticky=E)
ttk.Label(mainframe, text="%").grid(column=3, row=2, sticky=W)

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

iny_entry.focus()
root.bind('<Return>', calculate)

root.mainloop()
