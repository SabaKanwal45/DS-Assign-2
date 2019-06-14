from loadData import loadData
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import math


def plotTrainingData(train_data,tetas=[],titles=[],legends=[]):

    data = np.array(train_data)
    x_values = []
    y_values = []
    if(len(tetas) and len(tetas)<4):
        minimum = np.min(data[:, 0])
        maximum = np.max(data[:,0])
        x_values = np.arange(minimum, maximum, 0.5)
        y_values = - (tetas[0] + np.dot(tetas[1], x_values)) / tetas[2]

    elif(len(tetas)>4):
        minimum_1 = np.min(data[:, 1])-0.7
        maximum_1 = np.max(data[:,1])+0.7
        minimum_2 = np.min(data[:, 2])-0.7
        maximum_2 = np.max(data[:,2])+0.7
        delta = 0.025
        x_ = np.arange(minimum_1, maximum_1, delta)
        y_ = np.arange(minimum_2, maximum_2, delta)
        x1, x2 = np.meshgrid(x_,y_)
        contour = plt.contour(x1, x2,tetas[0]+tetas[1]*x1+tetas[2]*x2+
    tetas[3]* x1**2 +tetas[4]*x1*x2+tetas[5]*np.power(x2,2)+
    tetas[6]* x1**3+tetas[7]*np.power(x1,2)*np.power(x2,1)+tetas[8]*np.power(x1,1)*np.power(x2,2)+tetas[9]*np.power(x2,3)+
    tetas[10]* x1**4+tetas[11]*np.power(x1,3)*np.power(x2,1)+tetas[12]*np.power(x1,2)*np.power(x2,2)+tetas[13]*np.power(x1,1)*np.power(x2,3)+tetas[14]*np.power(x2,4)+
    tetas[15]* x1**5+tetas[16]*np.power(x1,4)*np.power(x2,1)+tetas[17]*np.power(x1,3)*np.power(x2,2)+tetas[18]*np.power(x1,2)*np.power(x2,3)+tetas[19]*np.power(x1,1)*np.power(x2,4)+tetas[20]*np.power(x2,5)+
    tetas[21]* x1**6+tetas[22]*np.power(x1,5)*np.power(x2,1)+tetas[23]*np.power(x1,4)*np.power(x2,2)+tetas[24]*np.power(x1,3)*np.power(x2,3)+tetas[25]*np.power(x1,2)*np.power(x2,4)+tetas[26]*np.power(x1,1)*np.power(x2,5)+tetas[27]*np.power(x2,6),[0],colors='g')


    data_P = [ex for ex in train_data if ex[2]==1.0]
    data_N = [ex for ex in train_data if ex[2]==0.0]
    exam1_P = [row[0] for row in data_P]
    exam2_P = [row[1] for row in data_P]
    exam1_N = [row[0] for row in data_N]
    exam2_N = [row[1] for row in data_N]
    scatter1 = plt.scatter(exam1_P,exam2_P,marker='+',color='k')
    scatter2 = plt.scatter(exam1_N,exam2_N,facecolors='y', edgecolors='y')
    plt.xlabel(titles[0])
    plt.ylabel(titles[1])
    if(len(x_values)):
        line = plt.plot(x_values, y_values)
        plt.legend([scatter1,scatter2, line[0]],legends,loc='upper right',ncol=1,fontsize=9)
    elif(len(tetas)>4):
        plt.legend([scatter1,scatter2,contour.collections[0]],legends,loc='upper right',ncol=1,fontsize=9)
    else:
        plt.legend([scatter1,scatter2],legends,loc='upper right',ncol=1,fontsize=9)
    #plt.text(15, -0.01, "Scatter Plot of training data", horizontalalignment='center', fontsize=12)
    plt.title(titles[2])
    plt.show()
#train_data = loadData('data2/ex2data1.txt')
#plotTrainingData(train_data)