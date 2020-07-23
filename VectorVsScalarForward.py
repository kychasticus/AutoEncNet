import pickle
import time
from utils import images2batches, get_random_batch
from train_autoenc_lite import EncDecNetLite

BATCH_SIZE = 20
UPDATES_NUM = 1000
IMG_SIZE = 15
D = 225  # IMG_SIZE*IMG_SIZE
P = 75  # D /// 3
LEARNING_RATE = 0.0001

# Load train data
images_train = pickle.load(open('images_train.pickle', 'rb'))
# Convert images to batching-friendly format
batches_train = images2batches(images_train)

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