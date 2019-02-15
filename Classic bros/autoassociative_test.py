import numpy as np
import matplotlib.pyplot as plt
import autoassociative

num_inputs = 10
inputs_shape = 30
zero = np.array([
    -1, 1, 1, 1,-1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
    -1, 1, 1, 1,-1
])
one = np.array([
    -1, 1, 1,-1,-1,
    -1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1
])
two = np.array([
     1, 1, 1, 1,-1,
    -1,-1,-1, 1,-1,
    -1,-1,-1, 1,-1,
    -1,-1, 1,-1,-1,
    -1, 1,-1,-1,-1,
    -1, 1, 1, 1, 1
])
three = np.array([
    -1, 1, 1, 1,-1,
    -1,-1,-1, 1,-1,
    -1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1,
    -1,-1,-1, 1,-1,
    -1, 1, 1, 1,-1
])
four = np.array([
    -1, 1,-1, 1,-1,
    -1, 1,-1, 1,-1,
    -1, 1, 1, 1,-1,
    -1,-1,-1, 1,-1,
    -1,-1,-1, 1,-1,
    -1,-1,-1, 1,-1
])
five = np.array([
    -1, 1, 1, 1, 1,
    -1, 1,-1,-1,-1,
    -1,-1, 1, 1,-1,
    -1,-1,-1,-1, 1,
    -1,-1,-1,-1, 1,
    -1, 1, 1, 1,-1
])
six = np.array([
    -1,-1, 1, 1,-1,
    -1, 1,-1,-1,-1,
    -1, 1, 1,-1,-1,
    -1, 1,-1, 1,-1,
    -1, 1,-1, 1,-1,
    -1,-1, 1,-1,-1
])
seven = np.array([
    -1, 1, 1, 1, 1,
    -1,-1,-1,-1, 1,
    -1,-1,-1, 1,-1,
    -1,-1,-1, 1,-1,
    -1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1
])
eight = np.array([
    -1, 1, 1, 1,-1,
    -1, 1,-1, 1,-1,
    -1, 1, 1, 1,-1,
    -1, 1,-1, 1,-1,
    -1, 1, 1, 1,-1,
    -1,-1,-1,-1,-1
])
nine = np.array([
    -1, 1, 1, 1, 1,
    -1, 1,-1,-1, 1,
    -1,-1, 1, 1, 1,
    -1,-1,-1,-1, 1,
    -1,-1,-1, 1,-1,
    -1,-1, 1,-1,-1
])

print("Shape of individual inputs : {}".format(nine.shape))
example = nine.reshape((6,5))
print("Shape after reshape = {}".format(example.shape))
print(example)
input_vector = np.stack([zero,one,two,three,four,five,six,seven,eight,nine],0).T
print("Inputs :")
print(input_vector)
cell = autoassociative.AutoAssociativeMemoryCell(inputs_shape,(6,5))
cell.compute_weights_pseudoinverse(input_vector)
cell.visualize_pattern(input_vector[:,7])
cell.compute_result(input_vector[:,7])
seven_copy = seven.copy()
seven_copy[4] = -1
seven_copy[27] = -1
cell.visualize_pattern(seven_copy)
cell.compute_result(seven_copy)