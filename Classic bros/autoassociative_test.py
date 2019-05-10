import numpy as np
import matplotlib.pyplot as plt
import patternutil
from autoassociative import *

SHAPE = (7,5)
dataset = np.load('./tools/better.npy')
num = 5
dataset = dataset[[0,8,18,22,24]]
inputs_num, input_shape = dataset.shape
dict = {i:j for i,j in zip(range(inputs_num),"ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
network = AutoAssociativeMemoryCell(input_shape,SHAPE)
network.compute_weights_pseudoinverse(dataset.T)
num_test = 5
# np.random.shuffle(dataset)
plt.imshow(network.weights,cmap='hot')
plt.show()
indices = np.arange(num_test)
# exit()
# normal case
for i in indices:
    pattern = dataset[i,:].T
    result = network.compute_result(pattern)
    patternutil.show_pattern_pair(pattern.T,result.T,SHAPE)

# occluded case
for i in indices:
    pattern = dataset[i,:]
    pattern = patternutil.add_noise(pattern,5).T
    result = network.compute_result(pattern)
    patternutil.show_pattern_pair(pattern.T,result.T,SHAPE)
# noise case
for i in indices:
    pattern = dataset[i,:]
    pattern = patternutil.occlude(pattern,3,SHAPE)
    pattern = pattern.T
    result = network.compute_result(pattern)
    patternutil.show_pattern_pair(pattern.T,result.T,SHAPE)