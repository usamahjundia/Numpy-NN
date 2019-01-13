import numpy as np

class Network(object):

    def __init__(self, sizes,debug = False):
        self.sizes = sizes
        self.debug = debug
        self.weights = [np.random.randn(k,j) for j,k in zip(sizes[:-1],sizes[1:])]
        self.biases = [np.random.randn(k,1) for k in sizes[1:]]
    
    def describe(self):
        print("="*20)
        print("A neural network with parameters :")
        print("Number of weights matrices : {}".format(len(self.weights)))
        print("Weights of size :")
        for weight in self.weights[:-1]:
            print(weight.shape,end=', ')
        print(self.weights[-1].shape)
        print("Number of biases : {}".format(len(self.biases)))
        print("Biases of size :")
        for bias in self.biases[:-1]:
            print(bias.shape,end=', ')
        print(self.biases[-1].shape)
        print("="*20)

    def sigmoid(self,x,isPrime=False):
        if not isPrime:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return self.sigmoid(x,isPrime=False) * (1 - self.sigmoid(x,isPrime=False))

    def feedforward(self,x):
        activation=x[:]
        for w,b in zip(self.weights,self.biases):
            activation = np.dot(w,activation) + b
            activation = self.sigmoid(activation)
        return activation

    def create_batched_data(self,x,y,batch_size):
        if len(x) != len(y):
            print()
        data_pair = [(np.expand_dims(a,0).T,np.array([[b]])) for a,b in zip(x,y[0])]
        np.random.shuffle(data_pair)
        batches = [data_pair[k:k+batch_size] for k in range(0,len(x),batch_size)]
        return batches

    
    def SGD(self,x,y,learning_rate,epochs,batch_size=32,x_test=None,y_test=None):
        test = False
        if x_test is not None and y_test is not None:
            n_test_data = len(x_test)
            test = True
        mini_batches = self.create_batched_data(x,y,batch_size)
        for i in range(epochs):
            for batch in mini_batches:
                self.update_mini_batch(batch,learning_rate)
    
    def update_mini_batch(self,batch,learning_rate):
        if self.debug:
            print("="*25)
            print("DEBUG STRING FOR UPDATE_MINI_BITCH")
        n = len(batch)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x,y in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x,y)
            nabla_w =[w+dw for w,dw in zip(nabla_w, delta_nabla_w)]
            nabla_b =[b+db for b,db in zip(nabla_b, delta_nabla_b)]
            # print(nabla_w)
            # print(nabla_b)
            if self.debug:
                print("NABLA_W SIZE :")
                for nw in nabla_w:
                    print(nw.shape,end=' ')
                print()
                print("NABLA_B SIZE :")
                for nw in nabla_b:
                    print(nw.shape,end=' ')
                print()
                print("END DEBUG STRING FOR UPDATE_MINI_BATCH")
                print("="*25)
        self.weights = [w - (learning_rate / n) * dw for w,dw in zip(self.weights,nabla_w)]
        self.biases = [b - (learning_rate / n) * db for b,db in zip(self.biases,nabla_b)]

    def backprop(self,x,y):
        if self.debug:
            print("="*25)
            print("DEBUG STRING FOR BACKPROP")
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        activations = [x]
        activation = x
        zs = []
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        if self.debug:
            print("Shapes of the activations : ",end="")
            for act in activations: print(act.shape,end= " ")
            print()
        error = self.loss_function(y,activation,prime=True)
        if self.debug:
            print("Output layer error : {} with shape {}".format(error,error.shape))
        delta_b[-1] = error
        delta_w[-1] = np.dot(error, activations[-2].T)
        for i in range(2,len(self.sizes)):
            error = np.dot(self.weights[-i+1].T, error) * self.sigmoid(zs[-i],isPrime=True)
            delta_b[-i] = error
            delta_w[-i] = np.dot(error,activations[-i-1].T)
        return delta_w, delta_b

    def loss_function(self,y_true,y_pred,prime=False):
        if self.debug:
            print("="*25)
            print("Begin Debug string for loss_function :")
            print("Inputs = {} , {}, {}".format(y_true,y_pred,prime))
            print("Inputs shape : {}   {}".format(y_true.shape,y_pred.shape))
        if prime:
            result = y_pred - y_true
            if self.debug:
                print("Will return : {}".format(result))
                print("Shape : {}".format(result.shape))
                print("End debug string for loss_function")
                print("="*25)
            return result
        return np.sum(np.nan_to_num(y_true * np.log(y_pred) + (1-y_true) * np.log((1- y_pred))))

