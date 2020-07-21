from __future__ import print_function
import numpy as np
import pickle
import time
from utils import get_loss, get_random_batch, images2batches, init_uniform, relu
from utils import init_reduce, relu_back

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

        self.inLayerForward(x)
        self.recLayerForward()
        self.linkLayerForward(x)
        self.outLayerForward()

        return self.y

    def inLayerForward(self, x):

        B_in = np.ones((BATCH_SIZE, 1)) @ self.b_in  # [20, 1] * [1, 75]
        self.a_in = x @ self.w_in + B_in  # [20, 225] * [225, 75] + [20, 75]
        self.z_in = relu(self.a_in)  # [20, 75]

        return self.z_in

    def recLayerForward(self):

        B_rec = np.ones((BATCH_SIZE, 1)) @ self.b_rec  # [20, 1] * [1, 75]
        self.a_rec = self.z_in @ self.w_rec + B_rec  # [20, 75] * [75, 75] + [20, 75]
        self.z_rec = relu(self.a_rec)  # [20, 75]

        #return self.z_rec

    def linkLayerForward(self, x):

        self.x_red = x @ self.ReduceMatrix  # [20, 225] * [225, 75]
        self.a_link = self.x_red @ self.w_link  # [20, 75] * [75, 75]
        self.z_link = self.a_link  # [20, 75]

        #return self.z_link

    def outLayerForward(self):

        B_out = np.ones((BATCH_SIZE, 1)) @ self.b_out  # [20, 1] * [1, 225]
        self.a_out = (self.z_link + self.z_rec) @ self.w_out + B_out  # [20, 75] * [75, 225]
        self.y = relu(self.a_out) # [20, 225]

        #return self.y

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


    def backprop(self, x):

        self.outLayerBackward(x)
        self.linkLayerBackward()
        self.recLayerBackward()
        self.inLayerBackward(x)
        self.apply_dw()


    def outLayerBackward(self, x):

        dLdy = (self.y - x) * 2 # [20,225] - [20,225]
        dYda_out = np.diag(relu_back(self.a_out)) # [225,225] ?? it seems we apply wrong matrice
        z = self.z_rec + self.z_link
        dA_outdZ_reclink = self.w_out.T # [225,75]

        self.dLdB_out = np.sum(dLdy @ dYda_out, axis=0)  # [20,225]@[225,225] ->[1,225]
        self.dLdZ_reclink = dLdy @ dYda_out @ dA_outdZ_reclink  # [20,225]@[225,225]@[225,75] ->[20,75]

        matrix = np.zeros((1, 75*225))
        for i in range(BATCH_SIZE):
            matrix += dLdy[i] @ dYda_out @ np.tile(z[i], (225, 225))  # [1,225]@[225,225]@[225,75*225]
        self.dL_dWout = matrix.reshape((225, 75)).T


    def linkLayerBackward(self):

        matrix = np.zeros((1, 75 * 75))
        for i in range(BATCH_SIZE):
            matrix += self.dLdZ_reclink[i] @ np.tile(self.x_red[i], (75, 75))  # [1,75]@[75,75*75]
        self.dL_dWlink = matrix.reshape((75, 75)).T


    def recLayerBackward(self):

        dZrec_dArec = np.diag(relu_back(self.a_rec))  # [75,75]?? it seems we apply wrong matrice
        dArec_dZin = self.w_rec.T  # [75,75]

        self.dL_dZin = self.dLdZ_reclink @ dZrec_dArec @ dArec_dZin  # [20,75][75,75][75,75]

    def inLayerBackward(self, x):
        dZin_dAin = np.diag(relu_back(self.a_in))  # [75,75]?? it seems we apply wrong

        self.dL_dBin = np.sum(self.dL_dZin @ dZin_dAin, axis=0)  # [20,75][75,75]

        matrix = np.zeros((1, 225 * 75))
        for i in range(BATCH_SIZE):
            matrix += self.dL_dZin[i] @ dZin_dAin @ np.tile(x[i], (75, 75))  # [1,75]@[75,75]@[75,225*75]
        self.dL_dWin = matrix.reshape((75, 225)).T


    def apply_dw(self):

        self.w_in -= LEARNING_RATE * self.dL_dWin  # [225,75]-[225,75]
        self.b_in -= LEARNING_RATE * self.dL_dBin  # [1,75]-[1,75]
        self.w_link -= LEARNING_RATE * self.dL_dWlink  # [75,75]-[75,75]
        self.w_out -= LEARNING_RATE * self.dL_dWout  # [75,225]-[75,225]
        self.b_out -= LEARNING_RATE * self.dLdB_out  # [1,225]-[1,225]


# Load train data
images_train = pickle.load(open('images_train.pickle', 'rb'))
# Convert images to batching-friendly format
batches_train = images2batches(images_train)
# Calculate the mean image
mean_image = np.mean(batches_train, axis=0)




# Create neural network for the timetest
neural_network_timetest = EncDecNetLite()
# Initialize weights for the timetest
neural_network_timetest.init()

# Measure the performance difference between Layer_in SCALAR and VECTOR
X_time_test = get_random_batch(batches_train, BATCH_SIZE)

start_time_vector = time.time()
neural_network_timetest.inLayerForward(X_time_test)
time_vector = time.time() - start_time_vector

start_time_scalar = time.time()
neural_network_timetest.inLayerForwardScalar(X_time_test)
time_scalar = time.time() - start_time_scalar
print(f"The Scalar Form  takes {time_scalar} seconds")
print(f"The Vector Form  takes {time_vector} seconds")
print(f"The Vector Form performs {time_scalar/time_vector} times faster the Scalar one")





# Create neural network for training
neural_network = EncDecNetLite()
# Initialize weights for training
neural_network.init()

# Main cycle
lossListTrain = []
for i in range(min(1, UPDATES_NUM)):
    # Get random batch for Stochastic Gradient Descent
    X_batch_train = get_random_batch(batches_train, BATCH_SIZE)
    X_demeaned = X_batch_train - np.ones((BATCH_SIZE,1)) @ mean_image.reshape((1,D))

    # Forward pass, calculate network''s outputs
    Y_batch = neural_network.forward(X_demeaned)

    # Calculate sum squared loss
    loss = get_loss(Y_batch, X_demeaned)
    lossListTrain.append(loss)

    print(f'Epoch {i}/{UPDATES_NUM}, Loss: {loss}')

    # Backward pass, calculate derivatives of loss w.r.t. weights
    dw = neural_network.backprop(i)

    # Correct neural network''s weights
    neural_network.apply_dw(dw)

#
# Load images_test.pickle here, run the network on it and show results here
#
