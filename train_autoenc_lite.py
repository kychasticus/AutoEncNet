from __future__ import print_function
import numpy as np
import pickle
import time
from utils import get_loss, get_random_batch, images2batches, init_uniform, relu, init_reduce


BATCH_SIZE = 20
UPDATES_NUM = 1000
IMG_SIZE = 15
D = 225 # IMG_SIZE*IMG_SIZE
P = 75 # D /// 3
LEARNING_RATE = 0.001


class EncDecNetLite():
    def __init__(self):
        super(EncDecNetLite, self).__init__()
        self.w_in = np.zeros((D, P)) # [225,75]
        self.b_in = np.zeros((1, P)) # [1,75]
        self.w_link = np.zeros((P, P)) # [75,75]
        self.w_out = np.zeros((P, D)) # [75,225]
        self.b_out = np.zeros((1, D)) # [1,225]
        self.w_rec = np.eye(P)        # [75,75]
        self.b_rec = np.zeros((1, P)) # [1,75]
        self.ReduceMatrix = np.zeros((D, P))

    def init(self):
        self.w_in = init_uniform(self.w_in)
        self.w_link = init_uniform(self.w_link)
        self.w_out = init_uniform(self.w_out)
        self.ReduceMatrix = init_reduce(self.ReduceMatrix)

    def forward(self, x):

        z_in = self.inLayerForward(x)
        z_rec = self.recLayerForward(z_in)
        z_link = self.linkLayerForward(x)
        y = self.outLayerForward(z_rec, z_link)

        return y

    def inLayerForward(self, x):

        B_in = np.ones((BATCH_SIZE, 1)) @ self.b_in  # [20, 1] * [1, 75]
        a_in = x @ self.w_in + B_in  # [20, 225] * [225, 75] + [20, 75]
        z_in = relu(a_in)  # [20, 75]

        return z_in

    def recLayerForward(self, z_in):

        B_rec = np.ones((BATCH_SIZE, 1)) @ self.b_rec  # [20, 1] * [1, 75]
        a_rec = z_in @ self.w_rec + B_rec  # [20, 75] * [75, 75] + [20, 75]
        z_rec = relu(a_rec)  # [20, 75]

        return z_rec

    def linkLayerForward(self, x):

        x_red = x @ self.ReduceMatrix  # [20, 225] * [225, 75]
        a_link = x_red @ self.w_link  # [20, 75] * [75, 75]
        z_link = a_link  # [20, 75]

        return z_link

    def outLayerForward(self, z_rec, z_link):

        B_out = np.ones((BATCH_SIZE, 1)) @ self.b_out  # [20, 1] * [1, 225]
        a_out = (z_link + z_rec) @ self.w_out + B_out  # [20, 75] * [75, 225]
        y = relu(a_out)

        return y

    def inLayerForwardScalar(self, x):

        a_in = np.zeros(shape=(BATCH_SIZE, P))

        for i in range(BATCH_SIZE):
            for p in range(P):
                xw = []

                for j in range(D):
                    xw.append(x[i, j] * self.w_in[j, p])

                a_in[i, p] = sum(xw) + self.b_in[0, p]

        z_in = a_in
        for k in range(a_in.shape[0]):
            for l in range(a_in.shape[1]):
                z_in[k, l] = relu(z_in[k, l])

        return z_in


    def backprop(self, some_args):
        #
        # Please, add backpropagation pass here
        #
        return 0 # dw


    def apply_dw(self, dw):
        #
        # Correct neural network''s weights
        #
        pass


# Load train data
images_train = pickle.load(open('images_train.pickle', 'rb'))
# Convert images to batching-friendly format
batches_train = images2batches(images_train)

# Create neural network
neural_network = EncDecNetLite()
# Initialize weights
neural_network.init()

# Measure the performance difference between Layer_in SCALAR and VECTOR
X_time_test = get_random_batch(batches_train, BATCH_SIZE)

start_time_vector = time.time()
neural_network.inLayerForward(X_time_test)
time_vector = time.time() - start_time_vector

start_time_scalar = time.time()
neural_network.inLayerForwardScalar(X_time_test)
time_scalar = time.time() - start_time_scalar
print(f"The Scalar Form  takes {time_scalar} seconds")
print(f"The Vector Form  takes {time_vector} seconds")
print(f"The Vector Form performs {time_scalar/time_vector} times faster the Scalar one")

# Main cycle
for i in range(UPDATES_NUM):
    # Get random batch for Stochastic Gradient Descent
    X_batch_train = get_random_batch(batches_train, BATCH_SIZE)

    # Forward pass, calculate network''s outputs
    Y_batch = neural_network.forward(X_batch_train)

    # Calculate sum squared loss
    loss = get_loss(Y_batch, X_batch_train)

    # Backward pass, calculate derivatives of loss w.r.t. weights
    dw = neural_network.backprop(some_args)

    # Correct neural network''s weights
    neural_network.apply_dw(dw)

#
# Load images_test.pickle here, run the network on it and show results here
#
