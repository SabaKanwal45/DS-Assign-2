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
        tetas.append(0)
    return tetas



def predict_instance(example, tetas):
    val = 0.0
    for index in range(len(tetas)):
        val += example[index]*tetas[index]
    return sigmoid(val)

def compute_cost(train_data,tetas,lamda):
    cost = 0
    for index in range(len(train_data)):
        h_x = predict_instance(train_data[index], tetas)
        label = train_data[index][len(train_data[0])-1]
        cost += label * np.log(h_x) + (1-label) * np.log(1-h_x)
    tetas_sum = 0
    for index in range(1,len(tetas)):
        tetas_sum += np.power(tetas[index],2)

    total_cost =  (1/len(train_data)) * (-cost + (lamda/2)*tetas_sum)

    return total_cost




def one_training_iteration(train_data,tetas,learning_rate,total_examples,lamda):
    temp_tetas = []
    for index in range(len(tetas)):
        diffrential_part = 0
        for ex in range(len(train_data)):
            h_x = predict_instance(train_data[ex], tetas)
            label = train_data[ex][len(train_data[0])-1]
            diffrential_part += (h_x - label) * train_data[ex][index]
        new_teta = 0
        if(index==0):
            new_teta = tetas[index] - learning_rate * (1/total_examples) * diffrential_part
        else:
            extra_term = 1 - (lamda/total_examples)*learning_rate
            new_teta = tetas[index] *extra_term - learning_rate * (1/total_examples) * diffrential_part
        temp_tetas.append(new_teta)
    return temp_tetas

def gradient_descent_logistic_regression(train_data,train_data_with_bias,learning_rate,lamda):
    # Add bias term in training data
    predicted_y = []
    tetas = initialize_tetas(len(train_data_with_bias[0])-1)
    if(len(train_data_with_bias)):
        tetas= one_training_iteration(train_data_with_bias,tetas,learning_rate,len(train_data_with_bias),lamda)
        print(tetas)
        for i in range(100):
            tetas= one_training_iteration(train_data_with_bias,tetas,learning_rate,len(train_data_with_bias),lamda)
            print("Iteration:  %(key1)s"%{'key1':i+1})
            cost = compute_cost(train_data_with_bias,tetas,0)
            print("Cost:  %(key1)s"%{'key1':cost})
        legends = ['y = 1','y = 0','Decision Boundary']
        titles = ["Microchip Test 1","Microchip Test 1","Scatter Plot of training data"]
        plotTrainingData(train_data,tetas,titles,legends)
        
    else:
        print("No training data")
    return tetas

def normalize_test_data(test_data,mean,std_dev):
    for i in range(len(test_data)):
        for att in range(len(test_data[0])-1):
            test_data[i][att] = (test_data[i][att] - mean[att])/std_dev[att]
    return test_data

def expand_features(train_data,power):
    expanded_data = []
    train_data = np.array(train_data)
    x1 = train_data[:,0]
    x2 = train_data[:,1]
    expanded_data.append( np.ones(len(train_data)))
    for i in range(1,power+1):
        temp = i
        j = 0
        for j in range(j,i+1,1):
            vect = np.power(x1,temp)*np.power(x2,j)
            expanded_data.append(vect)
            #print(" x1 " + str(temp) +" x2 " +str(j))
            temp = temp -1
    label = train_data[:,len(train_data[0])-1]
    expanded_data.append(label)
    matrix_data = np.array(expanded_data)
    matrix_data = matrix_data.transpose()
    return matrix_data


def main():
    #ex1data1.
    train_data = loadData('data2/ex2data2.txt')
    legends = ['y = 1','y = 0']
    titles = ["Microchip Test 1","Microchip Test 1","Scatter Plot of training data"]
    plotTrainingData(train_data,[],titles,legends)
    feature_vactor = expand_features(train_data,6)
    tetas= gradient_descent_logistic_regression(train_data,feature_vactor,1,1)
    tetas= gradient_descent_logistic_regression(train_data,feature_vactor,1,0)
    tetas= gradient_descent_logistic_regression(train_data,feature_vactor,1,100)
    #print(len(train_data))
    


main()