import numpy as np
def f(t, y): 
    return -t * y**2

def f_exacte(t):
    return 2 / (1 + t**2)

def z(t, y): 
    return -y + np.cos(t)

def z_exacte(t):
    return 0.5 * np.exp(-t) + 0.5 * np.cos(t) + 0.5 * np.sin(t)

def h(t, y): 
    return -2 * y

def h_exacte(t):
    return np.exp(-2 * t)