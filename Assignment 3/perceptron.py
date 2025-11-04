#-------------------------------------------------------------------------
# AUTHOR: Caitlyn Hue
# FILENAME: perceptron.py
# SPECIFICATION: Train Single-Layer and Multi-Layer Perceptron to classify optically recognized handwritten digits
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

#learning rate parameter
n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

#shuffle parameter
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

max_perceptron_accuracy = 0
max_MLP_accuracy = 0
for lr in n: #iterates over n

    for sh in r: #iterates over r

        algorithms = ['Perceptron', 'MLP']
        for alg in algorithms: #iterates over the algorithms

            #Create a Neural Network classifier
            if alg == 'Perceptron':
               clf = Perceptron(eta0=lr, shuffle=sh, max_iter=1000)    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            else:
                #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,shuffle = shuffle the training data, max_iter=1000
               clf = MLPClassifier(activation='logistic', learning_rate_init=lr, hidden_layer_sizes=(25,), shuffle=sh, max_iter=1000)
            
            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            correct = 0
            total = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])
                if prediction == y_testSample:
                    correct += 1
                total += 1

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            accuracy = correct/total

            if alg == 'Perceptron':
                if accuracy > max_perceptron_accuracy:
                    max_perceptron_accuracy = accuracy
                    print(f"Highest Perceptron accuracy so far: {max_perceptron_accuracy}, Parameters: learning rate={lr}, shuffle={sh}")
            else:
                if accuracy> max_MLP_accuracy:
                    max_MLP_accuracy = accuracy
                    print(f"Highest MLP accuracy so far: {max_MLP_accuracy}, Parameters: learning rate={r}, shuffle={sh}")