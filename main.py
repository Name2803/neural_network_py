import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import os
import time
print(plt.get_backend())

aaa = 0

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def showmat(self):
        print(self.who)
        print(self.wih)
        pass

    def train(self, inputs_list, targets_list):
        #преобразовать список выходных знвчений в двухмерный массив
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T

        #расшитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        #расчитатть исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        #рвсчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        #расчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        #ошибки скрытого слоя - это ошибки output_errors,
        #распределённые пропорционально весовым коэфициентам связей
        #и рекомбинирование на скрытых узлах
        hidden_errors = np.dot(self.who.T, output_errors)

        #обновить весовые коэффициенты связей между скрытым и выходными слями
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        #обновить весовые коэффициенты связей между входным и скрытым слоями
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    def query(self, inputs_list):

        #преобразовать список входных значений
        #в двумерный массив
        inputs = np.array(inputs_list, ndmin = 2).T

        #рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        #расчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        #рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        #рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        pass

    pass


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

print(np.random.rand(3, 3) - 0.5)

date_file = open("mnist_test.csv", encoding="utf-8")


print("wAit")
for line in date_file:
    date_list = line
    all_values = date_list.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99

    n.train(inputs, targets)
    pass
date_file.close()

#----show_waits--------
n.showmat()

#----test----
test = open("mnist_test.csv", encoding="utf-8")
list = test.readline(3)

all_values = list.split(',')
inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(inputs)

print(n.query(inputs))

