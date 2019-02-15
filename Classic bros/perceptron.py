import numpy as np
import os

class Perceptron(object):

    weightfile = 'weights.npy'
    biasfile = 'bias.npy'
    
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.random.randn(input_size)
        self.bias = np.random.rand()
    
    def hardlim(self,x):
        return np.greater_equal(x,0).astype(np.int32)
    
    def feedforward(self,input_vec):
        return self.hardlim(np.dot(input_vec,self.weights)+self.bias)

    def do_singular_update(self,x,error):
        if error > 0:
            self.weights += x
        elif error < 0:
            self.weights -= x
        self.bias += error

    def describe(self):
        print("A single-layer perceptron with parameters: ")
        for i,weight in enumerate(self.weights):
            print(" -> Weight for input {} : {}".format(i,weight))
        print(" -> Bias : {}".format(self.bias))

    def learn(self,data,label,epochs):
        for i in range(epochs):
            alltrue = True
            print("Epoch no : {}".format(i+1))
            for x,y in zip(data,label):
                pred = self.feedforward(x)
                error = y - pred
                if error != 0:
                    alltrue = False
                self.do_singular_update(x,error)
            if alltrue:
                print("Training completed at epoch {}".format(i+1))
                break

    def save_params(self,pathfile):
        if not os.path.exists(pathfile):
            os.mkdir(pathfile)
        np.save(os.path.join(pathfile,Perceptron.weightfile),self.weights)
        np.save(os.path.join(pathfile,Perceptron.biasfile),self.bias)
    
    def load_params(self,pathfile):
        if not os.path.exists(pathfile):
            print("Saved model {} was not found in the path".format(pathfile))
            return
        self.weights = np.load(os.path.join(pathfile,Perceptron.weightfile))
        self.bias = np.load(os.path.join(pathfile,Perceptron.biasfile))
        self.input_size = self.weights.shape[0]