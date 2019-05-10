import numpy as np
import matplotlib.pyplot as plt
import patternutil

data = np.load('./hasil.npy')
size = (7,5)

a = data[0]

patternutil.visualize_pattern(a,size)
a_noise = patternutil.add_noise(a,2)
patternutil.visualize_pattern(a_noise,size)
a_occlude = patternutil.occlude(a,0.2,size)
patternutil.visualize_pattern(a_occlude,size)