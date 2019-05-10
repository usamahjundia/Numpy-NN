from hopfield import *
import patternutil as pu
import numpy as np

SHAPE = (3,3)
alphabets = np.load('./threebythree.npy')
inputs_num, input_size = alphabets.shape
network = Hopfield(input_size)
num_used = inputs_num
alphabets = alphabets[:num_used]
alphabets = alphabets.T
network.compute_weights_pseudoinverse(alphabets)
#pick random choice of the alphabets
indices = np.arange(num_used)
np.random.shuffle(indices)
for i in range(inputs_num):
    choice = indices[i]
    input_vector = alphabets[:,choice]
    result = network.feedforward(input_vector)
    pu.show_pattern_pair(input_vector.T,result.T,SHAPE)
    np.random.shuffle(indices)
# exit()
# add noise
for i in range(inputs_num):
    choice = indices[i]
    input_vector = alphabets[:,choice]
    input_vector = pu.add_noise(input_vector.T,1).T
    result = network.feedforward(input_vector)
    pu.show_pattern_pair(input_vector.T,result.T,SHAPE)
np.random.shuffle(indices)
# occlude 25%
for i in range(inputs_num):
    choice = indices[i]
    input_vector = alphabets[:,choice]
    input_vector = pu.occlude(input_vector.T,0.1,SHAPE).T
    result = network.feedforward(input_vector)
    pu.show_pattern_pair(input_vector.T,result.T,SHAPE)