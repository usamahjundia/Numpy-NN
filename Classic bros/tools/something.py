import matplotlib.pyplot as plt
import numpy as np

SIZE = (7,5)
NUM = 26

arr = np.load('better.npy')

fig, ax = plt.subplots(5,6)

try:
    for i in range(5):
        for j in range(6):
            imag = arr[j+i*6].reshape(SIZE)
            ax[i,j].imshow(imag)
except IndexError:
    pass

plt.show()
