import matplotlib.pyplot as plt

# Polt function for 3D problems
def plot3d(t, sol, title="3D plot"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.plot(sol[:,0], sol[:,1], sol[:,2], lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()