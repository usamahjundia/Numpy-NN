from rnn import RNNCell
import numpy as np
cell = RNNCell(5,4,3,20)

x = np.random.randn(10,5)

y, states = cell.feedforward(x)

print(y.shape)
print(y)
print(states.shape)
print(states)

print(cell.predict(x))
y = np.zeros_like(y,dtype=np.int32)
idx = np.random.randint(0,3,y.shape[0])
print(idx)
y[np.arange(len(x)),np.array(idx)] = 1
print(y)
a,b,c = cell.backpropagation_through_time(x,y)

print(a)

print(b)

print(c)

xs = np.array([x])
ys = np.array([y])

loss = cell.calculate_loss(xs,ys)

print(loss)