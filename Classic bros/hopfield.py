import numpy as np
import numpy.linalg as npl

MAX_ITER = 10000

class Hopfield(object):

    def __init__(self,input_size):
        self.input_size = input_size
        self.weights = np.zeros((input_size,input_size))
    
    def compute_weights(self,input_vector):
        self.weights = input_vector.dot(input_vector.T)
    
    def compute_weights_pseudoinverse(self,input_vector):
        input_dim, num_patterns = input_vector.shape
        pinverse = npl.inv(input_vector.T.dot(input_vector))
        pinverse = pinverse.dot(input_vector.T)
        self.weights = input_vector.dot(pinverse)
    
    def satlin(self,input_vector):
        temp_vector = np.where(input_vector > 1, 1, input_vector)
        temp_vector = np.where(temp_vector < -1, -1, temp_vector)
        return temp_vector
    
    def feedforward(self,input_vector):
        current_state = input_vector.copy()
        prev_state = 0
        counter = 0
        while(not np.array_equal(current_state,prev_state)):
            if counter == MAX_ITER:
                print("NOT Converging after MAX_ITER")
                break
            prev_state = current_state.copy()
            current_state = self.weights.dot(current_state)
            current_state = self.satlin(current_state)
            counter+= 1
        return current_state