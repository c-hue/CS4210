#-------------------------------------------------------------------------
# AUTHOR: Caitlyn Hue
# FILENAME: naive_bayes.py
# SPECIFICATION: Read training set, output classification of each of the 10 instances
#                from the test set if the confidence is >=0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

outlook_map = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temp_map = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity_map = {"High": 1, "Normal": 2}
wind_map = {"Strong": 1, "Weak": 2}
class_map = {"Yes": 1, "No": 2}
#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
X = []
Y = []

for row in dbTraining:
    X.append([outlook_map[row[1]],
             temp_map[row[2]],
             humidity_map[row[3]],
             wind_map[row[4]]
             ])
    Y.append(class_map[row[5]])

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X,Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header of the solution
#--> add your Python code here
print(f"{'Day':<8}{'Outlook':<12}{'Temperature':<14}{'Humidity':<12}{'Wind':<10}{'PlayTennis':<12}{'Confidence'}")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in dbTest:
    instance = [
            outlook_map[row[1]],
            temp_map[row[2]],
            humidity_map[row[3]],
            wind_map[row[4]]
    ]
    prob = clf.predict_proba([instance])[0]
    prediction = clf.predict([instance])[0]
    class_pred = "Yes" if prediction == 1 else "No"
    confidence = max(prob)

    if confidence >= 0.75:
        print(f"{row[0]:<8}{row[1]:<12}{row[2]:<14}{row[3]:<12}{row[4]:<10}{class_pred:<12}{confidence:.2f}")


