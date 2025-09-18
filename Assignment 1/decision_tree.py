
#-------------------------------------------------------------------------
# AUTHOR: Caitlyn
# FILENAME: decision_tree.py
# SPECIFICATION: The program will read contact_lens.csv and output a decision tree.
# FOR: CS 4210- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.
#--> add your Python code here
#encode the original categorical training classes into numbers and add to the vector Y.
#--> add your Python code here

age_map = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
spectacle_map = {"Myope": 1, "Hypermetrope": 2}
astigmatism_map = {"No": 1, "Yes": 2}
tear_map = {"Reduced": 1, "Normal": 2}
class_map = {"No": 0, "Yes": 1}

for row in db:
    X.append([age_map[row[0]],
              spectacle_map[row[1]],
              astigmatism_map[row[2]],
              tear_map[row[3]]]
    )
    Y.append(class_map[row[4]])

#fitting the decision tree to the data using entropy as your impurity measure
#--> add your Python code here
#clf =
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X,Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'],
class_names=['Yes','No'], filled=True, rounded=True)
plt.show()