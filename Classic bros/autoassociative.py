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

    def visualize_pattern(self,input_vector):
        input_vector = np.where(input_vector > 0,1,0)
        input_vector = input_vector.astype(np.float32)
        input_vector = np.reshape(input_vector,self.actual_shape)
        plt.figure()
        plt.imshow(input_vector)
        plt.show()

    def compute_result(self,input_vector,visualize=True):
        print("======"*15)
        print("Debug string for compute_result")
        result = self.weights.dot(input_vector)
        print("Network output before hardlim :")
        print(result)
        result = self.hardlim(result)
        print("Network output after hardlim :")
        print(result)
        print("Final output of network after reshape :")
        result = result.reshape(self.actual_shape)
        print(result)
        if visualize:
            plt.figure()
            plt.imshow(result)
            plt.show()
        print("====="*15)