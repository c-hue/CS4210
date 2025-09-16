
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
    print("read good")

#encode the original categorical training features into numbers and add to the 4D array X.
#--> add your Python code here
# X =

#encode the original categorical training classes into numbers and add to the vector Y.
#--> addd your Python code here
# Y =

#fitting the decision tree to the data using entropy as your impurity measure
#--> addd your Python code here
#clf =

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'],
class_names=['Yes','No'], filled=True, rounded=True)
plt.show()