from __future__ import print_function
import numpy as np
import pickle
import time
import plotly.graph_objects as go
from utils import get_loss, get_random_batch, images2batches, init_uniform, relu, imshow
from utils import init_reduce, relu_back

BATCH_SIZE = 20
UPDATES_NUM = 1000
IMG_SIZE = 15
D = 225  # IMG_SIZE*IMG_SIZE
P = 75  # D /// 3
LEARNING_RATE = 0.0001


class EncDecNetLite():
    def __init__(self):
        super(EncDecNetLite, self).__init__()
        self.w_in = np.zeros((D, P))  # [225,75]
        self.w_link = np.zeros((P, P))  # [75,75]
        self.w_out = np.zeros((P, D))  # [75,225]
        self.w_rec = np.eye(P)        # [75,75]
        self.b_in = np.zeros((1, P))  # [1,75]
        self.b_out = np.zeros((1, D))  # [1,225]
        self.b_rec = np.zeros((1, P))  # [1,75]
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


    def outLayerBackward(self, x):

        dLdy = (self.y - x) * 2  # [20,225] - [20,225]
        dYda_out = relu_back(self.a_out)  # [20,225]
        z = self.z_rec + self.z_link  # [20,75]
        dA_outdZ_reclink = self.w_out.T  # [225,75]

        vectorB = np.zeros((1, 225))
        for j in range(BATCH_SIZE):
            vectorB += dLdy[j] @ np.diag(dYda_out[j])  # [1,225]@[225,225] -> [1,225]
        self.dL_dBout = vectorB
        self.dL_dZreclink = np.multiply(dLdy, dYda_out) @ dA_outdZ_reclink  # ([20,225]*[20,225])@[225,75] ->[20,75]

        matrixW = np.zeros((75, 225))
        for i in range(BATCH_SIZE):
            # matrixW += dLdy[i] @ np.diag(dYda_out[i]) @ np.tile(z[i], (225, 225))  # [1,225]@[225,225]@[225,75*225]
            matrixW += z[i].reshape((75, 1)) @ dLdy[i].reshape((1, 225)) @ np.diag(dYda_out[i])  # [75*1]@[1,225]@[225,225]
        self.dL_dWout = matrixW


    def linkLayerBackward(self):

        matrix = np.zeros((75, 75))
        for i in range(BATCH_SIZE):
            # matrix += self.dL_dZreclink[i] @ np.tile(self.x_red[i], (75, 75))  # [1,75]@[75,75*75]
            matrix += self.x_red[i].reshape((75, 1)) @ self.dL_dZreclink[i].reshape((1, 75))  # [75,1][1,75]
        self.dL_dWlink = matrix


    def recLayerBackward(self):

        dZrec_dArec = relu_back(self.a_rec)  # [20,75]
        dArec_dZin = self.w_rec.T  # [75,75]

        self.dL_dZin = np.multiply(self.dL_dZreclink, dZrec_dArec) @ dArec_dZin  # ([20,75]*[20,75])[75,75]

    def inLayerBackward(self, x):
        dZin_dAin = relu_back(self.a_in)  # [20,75]

        vectorB = np.ones((1, 75))
        for j in range(BATCH_SIZE):
            vectorB += self.dL_dZin[j] @ np.diag(dZin_dAin[j])  # [1,75]@[75,75]->[1,75]
        self.dL_dBin = vectorB

        matrixW = np.zeros((225, 75))
        for i in range(BATCH_SIZE):
            # matrixW += self.dL_dZin[i] @ np.diag(dZin_dAin[i]) @ np.tile(x[i], (75, 75))  # [1,75]@[75,75]@[75,225*75]
            matrixW += x[i].reshape((225, 1)) @ self.dL_dZin[i].reshape((1, 75)) @ np.diag(dZin_dAin[i])  # [225,1][1,75]@[75,75]
        self.dL_dWin = matrixW


    def apply_dw(self):

        self.w_in = self.w_in - LEARNING_RATE * self.dL_dWin  # [225,75]-[225,75]
        self.b_in = self.b_in - LEARNING_RATE * self.dL_dBin  # [1,75]-[1,75]
        self.w_link = self.w_link - LEARNING_RATE * self.dL_dWlink  # [75,75]-[75,75]
        self.w_out = self.w_out - LEARNING_RATE * self.dL_dWout  # [75,225]-[75,225]
        self.b_out = self.b_out - LEARNING_RATE * self.dL_dBout  # [1,225]-[1,225]


# Load train data
images_train = pickle.load(open('images_train.pickle', 'rb'))
# Convert images to batching-friendly format
batches_train = images2batches(images_train)
# Calculate the mean image
mean_image = np.mean(batches_train, axis=0).reshape((1, D))
# std_image = np.std(batches_train.reshape((1, batches_train.shape[0]*D)))

# Create neural network for training
neural_network = EncDecNetLite()
# Initialize weights for training
neural_network.init()

# Main cycle
neural_start = time.time()
lossListTrain = []
for i in range(UPDATES_NUM):
    # Get random batch for Stochastic Gradient Descent
    X_batch_train = get_random_batch(batches_train, BATCH_SIZE)
    X_demeaned = X_batch_train - np.ones((BATCH_SIZE, 1)) @ mean_image
    X_scaled = X_demeaned / 255

    # Forward pass, calculate network''s outputs
    Y_batch = neural_network.forward(X_scaled)

    # Calculate sum squared loss
    loss = get_loss(Y_batch, X_scaled)
    lossListTrain.append(loss)

    print(f'Epoch {i}/{UPDATES_NUM}, Loss: {loss}, TimeLapsed: {time.time() - neural_start}')

    # Backward pass, calculate derivatives of loss w.r.t. weights
    dw = neural_network.backprop(X_scaled)

    # Correct neural network''s weights
    neural_network.apply_dw()


# Loss Curve
lossFigure = go.Figure()
batchCoordinate = []
for i in range(UPDATES_NUM):
    batchCoordinate.append(i)
lossFigure.add_trace(go.Scatter(x=batchCoordinate, y=lossListTrain))
lossFigure.update_layout(title='Mean Squared Loss for each Epoch')
lossFigure.show()

#
# Load images_test.pickle here, run the network on it and show results here
#
# Load train data
images_test = pickle.load(open('images_test.pickle', 'rb'))
# Convert images to batching-friendly format
batches_test = images2batches(images_test)
# Calculate the mean image
#mean_image_test = np.mean(batches_test, axis=0)
batches_test_demeaned = batches_test - np.ones((batches_test.shape[0], 1)) @ mean_image
batches_test_scaled = batches_test_demeaned / 255

y_test_demeaned = neural_network.forward(batches_test_scaled)
y_test = (y_test_demeaned * 255) + (np.ones((batches_test.shape[0], 1)) @ mean_image)


for l in range(BATCH_SIZE):
    imshow(batches_test[l].reshape((IMG_SIZE, IMG_SIZE)))
    imshow(y_test[l].reshape((IMG_SIZE, IMG_SIZE)))
