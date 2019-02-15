from rnn import RNNCell
import numpy as np
cell = RNNCell(1,4,3,20)

x = np.random.randn(10,1)

y, states = cell.feedforward(x)

print(y.shape)
print(y)
print(states.shape)
print(states)

print(cell.predict(x))