import neuralnet
import numpy as np

network = neuralnet.Network([2,3,4,3,2,1])
network.describe()
x = np.array([[0,0],[0,1],[1,0],[1,1]]).T
print("Input shape : {}".format(x.shape))
y = network.feedforward(x)
print(y)
print(y.shape)

dw,db = network.backprop(x,y)
print("Shapes of grads = ")
print("Weights :")
for w in dw:
    print("   - {}".format(w.shape))

print("Biases :")
for b in db:
    print("   - {}".format(b.shape))

print("="*50)
print("="*50)
print("="*50)
print("Begin testing for training")
print("Initializing new network")

net = neuralnet.Network([2,4,1])
net.describe()
data = np.array([[0,0],[0,1],[1,0],[1,1]])
label = np.array([[0,1,1,0]])
print("Initial outputs of network :")
print(net.feedforward(data.T))
print("Expected output of the network :")
print(label)
net.SGD(data,label,10e-3,200000,1)
print("After training :")
print(net.feedforward(data.T))
net.describe()