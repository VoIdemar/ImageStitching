from math import pi
import numpy as np

def gaussian(x, y, sigma):
    return np.exp(-(x**2 + y**2)/(2.0*sigma**2))/(2.0*pi*sigma**2)

def sgn(x):
    return (0 if x == 0 else
            1 if x > 0 else
            -1)