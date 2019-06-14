import numpy as np
import random
from loadData import loadData
from common import add_bias_term_in_data
from sigmoid import sigmoid
from featureNormaliz import normalize_features
from plotTrainingData import plotTrainingData
import math


def initialize_tetas(size):
    tetas = []
    for i in range(size):
        tetas.append(0.1)
    return tetas



def predict_instance(example, tetas):
    val = 0.0
    for index in range(len(tetas)):
        val += example[index]*tetas[index]
    return sigmoid(val)

def compute_cost(train_data,tetas):
    cost = 0
    for index in range(len(train_data)):
        h_x = predict_instance(train_data[index], tetas)
        label = train_data[index][len(train_data[0])-1]
        cost += label * np.log(h_x) + (1-label) * np.log(1-h_x)
    return (-1/len(train_data)) * cost




def one_training_iteration(train_data,tetas,learning_rate,total_examples):
    temp_tetas = []
    for index in range(len(tetas)):
        diffrential_part = 0
        for ex in range(len(train_data)):
            h_x = predict_instance(train_data[ex], tetas)
            label = train_data[ex][len(train_data[0])-1]
            diffrential_part += (h_x - label) * train_data[ex][index]
        new_teta = tetas[index] - learning_rate * (1/total_examples) * diffrential_part
        temp_tetas.append(new_teta)
    return temp_tetas

def gradient_descent_logistic_regression(train_data,learning_rate):
    # Add bias term in training data
    train_data,mean,std_dev = normalize_features(train_data)
    train_data_with_bias = add_bias_term_in_data(train_data)
    predicted_y = []
    tetas = initialize_tetas(len(train_data_with_bias[0])-1)
    if(len(train_data_with_bias)):
        tetas= one_training_iteration(train_data_with_bias,tetas,learning_rate,len(train_data_with_bias))
        print(tetas)
        for i in range(1000):
            tetas= one_training_iteration(train_data_with_bias,tetas,learning_rate,len(train_data_with_bias))
            print("Iteration:  %(key1)s"%{'key1':i+1})
            cost = compute_cost(train_data_with_bias,tetas)
            print("Cost:  %(key1)s"%{'key1':cost})
        legends = ['Admitted','Not Admitted','Decision Boundary']
        titles = ["Exam 1 Score","Exam 2 Score","Traning Data with decision boundary"]
        plotTrainingData(train_data,tetas,titles,legends)
        
    else:
        print("No training data")
    return tetas,mean,std_dev

def normalize_test_data(test_data,mean,std_dev):
    for i in range(len(test_data)):
        for att in range(len(test_data[0])-1):
            test_data[i][att] = (test_data[i][att] - mean[att])/std_dev[att]
    return test_data

def main():
    #ex1data1.
    train_data = loadData('data2/ex2data1.txt')
    legends = ['Admitted','Not Admitted']
    titles = ["Exam 1 Score","Exam 2 Score","Scatter Plot of training data"]
    plotTrainingData(train_data,[],titles,legends)
    #train_data = [[0,0,0],[0,1,1],[1,0,1],[1,1,1]]
    tetas,mean,std_dev = gradient_descent_logistic_regression(train_data,0.8)
    test_data = [[45,85,1]]
    test_data = normalize_test_data(test_data,mean,std_dev)
    test_data = add_bias_term_in_data(test_data)
    for i in range(len(test_data)):
        predicted_value = predict_instance(test_data[i],tetas)
        print("Probability of Test Example is %(key2)s"%{'key1':i+1,'key2':predicted_value})
    #print(len(train_data))
    


main()