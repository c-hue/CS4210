#-------------------------------------------------------------------------
# AUTHOR: Caitlyn Hue
# FILENAME: knn.py
# SPECIFICATION: Compute LOO-CV error rate for a 1NN classifier on spam/ham classification task.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

#Loop your data to allow each instance to be your test set
errors = 0
for i in db:
    X = []
    Y = []

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    for row in db:
        if row != i:
            X.append([float(val) for val in row[0:20]])
            if row[20] == "spam":
                Y.append(1)
            else:
                Y.append(0)

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(val) for val in i[0:20]]
    true_class = 1 if i[20] == "spam" else 0

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X,Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != true_class:
        errors += 1

#Print the error rate
#--> add your Python code here
error_rate = errors / len(db)
print(f"Error rate for 1NN: {error_rate:.3f}")