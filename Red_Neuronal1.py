import os
print(os.__file__)

import numpy as np
from matplotlib import pyplot as plt
import time
from matplotlib.animation import FuncAnimation,FFMpegFileWriter

from sklearn.datasets import make_circles

# In[2]:


# PROBLEM
n = 500
p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis]

def plot_circle():
    plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
    plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
    plt.axis("equal")
    plt.show()
    
#plot_circle()


# In[3]:


# FUNCIONES DE ACTIVACION
sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x))

relu = lambda x: np.maximum(0, x)

def plot_activation_function():
    _x = np.linspace(-5, 5, 100)
    plt.plot(_x, relu(_x))


# In[4]:


# CAPA DE LA RN
class NNLayer:
    num_connections = 0
    num_neurons = 0
    activation_function = sigm

    def __init__(self, num_conections, num_neurons, activation_function):
        self.num_neurons = num_neurons
        self.num_connections = num_conections
        self.activation_function = activation_function

        self.bias = np.random.rand(1, num_neurons) * 2 - 1
        self.weights = np.random.rand(num_conections, num_neurons) * 2 - 1

    def info(self, l):
        print('Capa {} - N:{} C:{}'.format(l, self.num_neurons, self.num_connections))

    def val(self,n):
        return self.bias[0][n]

# In[5]:

# RED N
class NN():

    def __init__(self):
        self.layer = []
        self.layers = 0
        self.topology = []
        self.hidden_layers = 0
        self.out = [(None, X)]

    def from_topology(self, topology: object, act_f: object) -> object:
        self.topology = topology
        self.layers = len(topology)
        self.hidden_layers = self.layers - 2

        # add layers
        for l, t in enumerate(topology[:-1]):
            print('Add layer {2:d}: con({0:2d}) neurons({1:2d})'.format(topology[l], topology[l + 1], l))
            self.layer.append(NNLayer(topology[l], topology[l + 1], act_f))

        #todo: add last layer
        self.layer.append(NNLayer(topology[-1], 1, act_f))

    def from_layers(self, num_inputs: int = 0, layers: object = []) -> object:
        self.layers = len(layers)
        self.hidden_layers = self.layers - 1
        self.topology = [num_inputs]
        for n, l in enumerate(layers):
            print('Add layer {2:d}: con({0:2d}) neurons({1:2d})'.format(l.num_connections, l.num_neurons, n))
            self.layer.append(l)
            self.topology.append(l.num_neurons)

    def fit(self,X):
        self.out = [(None, X)]

        # Forward pass
        for layer in nn.layer:
            z = self.out[-1][1] @ layer.weights + layer.bias
            a = layer.activation_function[0](z)

            self.out.append((z, a))

        return self.out[-1][1]

    def train(self,X,Y,lr=0.5):
        # Backward pass
        deltas = []
        self.fit(X)
        for l in reversed(range(0, nn.layers - 1)):
            z = self.out[l + 1][0]
            a = self.out[l + 1][1]

            layer = self.layer[l]

            if l == nn.layers - 2:
                deltas.insert(0, l2_cost[1](a, Y) * layer.activation_function[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * layer.activation_function[1](a))

            _W = layer.weights

            # Gradient descent
            layer.bias = layer.bias - np.mean(deltas[0], axis=0, keepdims=True) * lr
            layer.weights = layer.weights - self.out[l][1].T @ deltas[0] * lr

        return self.out[-1][1]

    def closs(self,Y):
        return l2cost[0](out[-1][1],Y)

    def draw_neural_net(self, ax, left, right, bottom, top):
        '''
        Draw a neural network cartoon using matplotilb.

        :usage:
            >>> fig = plt.figure(figsize=(12, 12))
            >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

        :parameters:
            - ax : matplotlib.axes.AxesSubplot
                The axes on which to plot the cartoon (get e.g. by plt.gca())
            - left : float
                The center of the leftmost node(s) will be placed here
            - right : float
                The center of the rightmost node(s) will be placed here
            - bottom : float
                The center of the bottommost node(s) will be placed here
            - top : float
                The center of the topmost node(s) will be placed here
            - layer_sizes : list of int
                List of layer sizes, including input and output dimensionality
        '''
        v_spacing = (top - bottom) / float(max(self.topology))
        h_spacing = (right - left) / float(len(self.topology) - 1)
        # Nodes
        for n, layer_size in enumerate(self.topology):
            layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
            for m in range(layer_size):
                x, y = (n * h_spacing + left, layer_top - m * v_spacing)
                print("c",n," n",m)
                color="w"
                val = 0
                if n > 0:
                    val = self.layer[n-1].val(m)
                    if val > 0:
                        color="lime"
                    else:
                        color="tomato"

                circle = plt.Circle((x,y), v_spacing / 4., color=color, ec='k', zorder=4)
                ax.add_artist(circle)

                label = plt.text(x,y,"{2:.3f}".format(n,m,val),fontsize=8, ha="center", zorder=5)
                ax.add_artist(label)
        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(self.topology[:-1], self.topology[1:])):
            layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
            layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                      [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='silver')
                    ax.add_artist(line)
        return ax

# In[16]:


# FUNCION DE ENTRENAMIENTO

topology = [p, 16, 8, 1]

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: (Yp - Yr))

def train(nn, X, Y, l2_cost, lr=0.25, train=True):
    out = [(None, X)]

    # Forward pass
    for layer in nn.layer[:-1]:
        z = out[-1][1] @ layer.weights + layer.bias
        a = layer.activation_function[0](z)

        out.append((z, a))

    if train:

        # Backward pass
        deltas = []

        for l in reversed(range(0, nn.layers - 1)):
            z = out[l + 1][0]
            a = out[l + 1][1]

            if l == nn.layers - 2:
                deltas.insert(0, l2_cost[1](a, Y) * nn.layer[l].activation_function[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * nn.layer[l].activation_function[1](a))

            _W = nn.layer[l].weights

            # Gradient descent
            nn.layer[l].bias = nn.layer[l].bias - np.mean(deltas[0], axis=0, keepdims=True) * lr
            nn.layer[l].weights = nn.layer[l].weights - out[l][1].T @ deltas[0] * lr

    return out[-1][1]


from IPython.display import clear_output
def visual_plot():
    # VISUALIZACIÓN Y TEST

    import time
    

    loss = []

    for i in range(1000):

        # Entrenemos a la red!
        pY = train(nn, X, Y, l2_cost, lr=0.05)

        if i % 25 == 0:

            print(pY)

            loss.append(l2_cost[0](pY, Y))

            res = 100

            _x0 = np.linspace(-1.5, 1.5, res)
            _x1 = np.linspace(-1.5, 1.5, res)

            _Y = np.zeros((res, res))

            for i0, x0 in enumerate(_x0):
                for i1, x1 in enumerate(_x1):
                    _Y[i0, i1] = train(nn, np.array([[x0, x1]]), Y, l2_cost, train=False)[0][0]

            plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
            plt.axis("equal")

            plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
            plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")

            clear_output(wait=True)
            plt.show()

            plt.plot(range(len(loss)), loss)

            time.sleep(0.1)
            
            plt.show()
            # print(pY)


from IPython.display import clear_output


def visual_plot2():
    # VISUALIZACIÓN Y TEST

    import time

    loss = []

    for i in range(1000):

        # Entrenemos a la red!
        pY = nn.train(X, Y, lr=0.05)

        if i % 25 == 0:

            #print(pY)

            cost = l2_cost[0](pY, Y)
            loss.append(cost)
            print(i,cost)

            res = 100

            _x0 = np.linspace(-1.5, 1.5, res)
            _x1 = np.linspace(-1.5, 1.5, res)

            _Y = np.zeros((res, res))

            for i0, x0 in enumerate(_x0):
                for i1, x1 in enumerate(_x1):
                    _Y[i0, i1] = nn.fit(np.array([[x0, x1]]))[0][0]

            plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
            plt.axis("equal")

            plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
            plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")

            clear_output(wait=True)
            #plt.show()

            plt.plot(range(len(loss)), loss)

            time.sleep(0.1)

            plt.show()

def vis_train(n,j):
    for i in range(n):
        # Entrenemos a la red!
        pY = nn.train(X, Y, lr=0.05)
        if i % j == 0:
            plt.clf()
            nn.draw_neural_net(ax, .1, .9, .1, .9)
            plt.show()
            time.sleep(0.1)


zz = NN()
zz.from_topology([p, 4, 8, 1], sigm)

l1 = NNLayer(2, 4, sigm)
l2 = NNLayer(4, 8, sigm)
l3 = NNLayer(8, 1, sigm)

nn = NN()
nn.from_layers(2, [l1, l2, l3])

#from Draw_NN import draw_neural_net

fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
#nn.draw_neural_net(ax, .1, .9, .1, .9)
vis_train(1000,100)
#ani = FuncAnimation(fig, animate, frames=1000,
#                    blit=True, interval = 2.5,repeat=False)

plt.show()

#visual_plot2()