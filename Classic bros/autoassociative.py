import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

class AutoAssociativeMemoryCell():
    
    def __init__(self,input_size,actual_shape):
        self.weights = np.zeros((input_size,input_size))
        self.actual_shape = actual_shape

    def compute_weights(self,input_vector):
        self.weights = input_vector.dot(input_vector.T)
    
    def compute_weights_pseudoinverse(self,input_vector):
        pinverse = npl.inv(input_vector.T.dot(input_vector))
        pinverse = pinverse.dot(input_vector.T)
        self.weights = input_vector.dot(pinverse)

    def hardlim(self,input_vector):
        return np.where(input_vector >= 0,1,-1)

    def compute_result(self,input_vector):
        result = np.matmul(self.weights,input_vector)
        result = self.hardlim(result)
        return result

