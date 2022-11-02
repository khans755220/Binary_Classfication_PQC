import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import tensorflow as tf
import math

num_modes = 2


#select quantum device (simulator)
dev = qml.device("strawberryfields.fock", wires = num_modes, cutoff_dim = 2)

#quantum circuit used for binary classification
def layer(v):
    qml.Beamsplitter(v[0], v[1], wires = [0,1])
    qml.Rotation(v[2], wires=0)
    qml.Rotation(v[3], wires=1)
    qml.Squeezing(v[4], 0.0, wires=0)
    qml.Squeezing(v[5], 0.0, wires=1)
    qml.Beamsplitter(v[6], v[7], wires=[0,1])
    qml.Rotation(v[8], wires=0)
    qml.Rotation(v[9], wires=1)
    qml.Displacement(v[10], 0.0, wires=0)
    qml.Displacement(v[11], 0.0, wires=1)
    qml.Kerr(v[12], wires=0)
    qml.Kerr(v[13], wires=1)


@qml.qnode(dev)

#neural network function containing state preperation
def quantum_neural_net(var,x):
    
    qml.Displacement(x[0], 0.0, wires=0)
    qml.Displacement(x[1], 0.0, wires=1)
    for v in var:
        layer(v)
    
    out1 = qml.expval(qml.X(0))
    return out1

#function to calculate squareloss and accuracy
def square_loss(labels, predictions):
    loss = 0
    accuracy = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l-p) ** 2

        if p > 0 and l == 1:
            accuracy += 1

        elif p < 0 and l == -1:
            accuracy += 1

    loss = loss / len(labels)
    accuracy = accuracy / len(labels)

    training_acc.append(accuracy)
    
    print(accuracy)
    
    return loss

#cost function used to train model
def cost(var, features, labels):
    
    preds = [quantum_neural_net(var, x) for x in features]

    return square_loss(labels, preds)

import matplotlib.pyplot as plt
import pandas as pd




#load dataset values
df = pd.read_csv('sinegen.txt', sep=' ', header=None)
X = df[0]
Y = df[1]
Z = df[2]

tupleList = tuple(zip(X, Y))

#create datasets
plt.figure()
plt.scatter(X, Y, c = Z, cmap = plt.cm.gray_r, edgecolor='black')
plt.xlabel("x", fontsize=18)
plt.ylabel("f(x)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.tick_params(axis="both", which="minor", labelsize=16)
plt.show()

np.random.seed(0)
num_layers = 4

#weights
var_init = 0.05 * np.random.randn(num_layers, 14, requires_grad=True)

opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)
var = var_init

training_acc = []
training_cost = []

#training model and optimizing results
for it in range(20):
    (var, _, _), _cost = opt.step_and_cost(cost, var, tupleList, Z)
    print("Iter: {:5d} | Cost: {:0.7f} ".format(it, _cost))
    training_cost.append(_cost)
    print(training_acc)

vals = [(i, j) for i in np.linspace(-1, 1, 50) for j in np.linspace(-1,1,50)]

new_X = []
new_Y = []

for i in range(len(vals)):
    
    #print(vals[i][0])
    new_X.append(vals[i][0])
    new_Y.append(vals[i][1])


#print(new_X)
#print(new_Y)

#output of model
predictions = [quantum_neural_net(var, x_) for x_ in vals]



class_A = [-0.5, 0.5]
class_B = [0.5, -0.5]

preds_accuracy = 0


#calculate accuracy of predictions
for i in range(len(new_X)):
    distanceA = math.sqrt(((class_A[0]-new_X[i])**2)+((class_A[1]-new_Y[i])**2))
    distanceB = math.sqrt(((class_B[0]-new_X[i])**2)+((class_B[1]-new_Y[i])**2))
    
    if(distanceA<distanceB and predictions[i]<0):
        preds_accuracy += 1

    elif(distanceA>distanceB and predictions[i]>0):
        preds_accuracy += 1

    #from IPython import embed; embed()

print("Final accuracy: ", preds_accuracy/2500)


#output results
plt.figure()
plt.scatter(new_X, new_Y, c = predictions, cmap = "RdBu")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tick_params(axis="both", which="major")
plt.tick_params(axis="both", which="minor")
plt.show()

plt.plot(np.linspace(1, 20, 20), training_acc)
plt.title("Accuracy of Training")
plt.xlabel("steps")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, 20, step=5))
plt.show()


plt.plot(np.linspace(1, 20, 20), training_cost)
plt.title("Cost of Training")
plt.xlabel("steps")
plt.ylabel("Squareloss")
plt.xticks(np.arange(1, 20, step=5))
plt.show()
