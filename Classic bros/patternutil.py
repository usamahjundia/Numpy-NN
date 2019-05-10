import numpy as np
import matplotlib.pyplot as plt

def add_noise(pattern, number,rows=None,cols=None):
    pattern_copy = pattern.copy()
    length = pattern.shape[0]
    indices = np.arange(length)
    np.random.shuffle(indices)
    for i in indices[:number]:
        pattern_copy[i] *= -1
    return pattern_copy

def occlude(pattern, num,shape):
    pattern_copy = pattern.copy()
    length = pattern.shape[0]
    rows, cols = shape
    amount = num
    amount *= cols
    amount = length - amount
    pattern_copy[amount:] = -1
    return pattern_copy

def visualize_pattern(pattern,shape):
    pattern_copy = pattern.reshape(shape)
    plt.imshow(pattern_copy)
    plt.show()

def show_pattern_pair(pattern1,pattern2,shape):
    p1 = pattern1.reshape(shape)
    p2 = pattern2.reshape(shape)
    fig,axes = plt.subplots(1, 2)
    axes[0].imshow(p1)
    axes[1].imshow(p2)
    plt.show()
    
