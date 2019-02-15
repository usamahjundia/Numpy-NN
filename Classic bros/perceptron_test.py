import perceptron
import numpy as np

ptron = perceptron.Perceptron(2)
print("Model init :")
ptron.describe()

data=np.array([[0,0],[0,1],[1,0],[1,1]])
label = np.array([0,0,0,1])

print("Initial predictions :")
print(ptron.feedforward(data))

print("Training begin")
ptron.learn(data,label,10)
print("Training finished")

print(ptron.feedforward(data))
ptron.describe()
