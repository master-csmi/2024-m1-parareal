import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from lorenz import *
from euler import *
from plot import *

#initial conditions
u0_1 = np.array([1.,1.,1.])
u0_2 = np.array([1.01, 1.01, 1.01])

# params
tspan = np.array([0.,60.])
sigma, rho, beta = 10, 28, 8/3
Nh = 10000

lorenz_sys = lambda _, state :lorenz(_, state, sigma, rho, beta)

t, u1 = feuler(lorenz_sys, tspan, u0_1, Nh)
t, u2 = feuler(lorenz_sys, tspan, u0_2, Nh)

# Extract the solution
x1, y1, z1 = u1.T
x2, y2, z2 = u2.T

# Create a figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim((-20, 20))
ax.set_ylim((-30, 30))
ax.set_zlim((0, 50))

# Create a line object for the animation
line1, = ax.plot([], [], [], lw=1)
line2, = ax.plot([], [], [], lw=1)

# Initialization function for the animation
def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    return line1,line2

# Animation update function
def update(num):
    line1.set_data(x1[:num], y1[:num])
    line1.set_3d_properties(z1[:num])
    line2.set_data(x2[:num], y2[:num])
    line2.set_3d_properties(z2[:num])
    return line1,line2

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t)//4, init_func=init, blit=True, interval=1e-10)

# Display the animation
plt.show()