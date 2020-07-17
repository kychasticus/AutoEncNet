import numpy as np
import matplotlib.pyplot as plt


def diff_numpy(a, b, msg=None):
    """Shows differences between two tensors"""
    if a.shape != b.shape:
        print('Wrong shape!')
        print(a.shape)
        print(b.shape)
    else:
        diff = (np.sum(a - b))**2
        if msg:
            print('%s diff = %1.6f' % (msg, diff.item()))
        else:
            print('diff = %1.6f' % diff.item())


def images2batches(images):
    """Converts images to convenient for batching form"""
    ndata, img_size, _ = images.shape
    return np.reshape(images, (ndata, img_size*img_size))


def imshow(img):
    """Show image using matplotlib"""
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def init_uniform(W):
    """Makes iniform initialization of weight matrix (please, use
    numpy.random.uniform function or similar"""

    n, m = W.shape[0], W.shape[1]
    normFactor = np.sqrt(6)/np.sqrt(n+m)
    weightArray = np.random.uniform(-normFactor, normFactor, n*m)

    return weightArray.reshape(n,m)

def init_reduce(R):

    P = R.shape[1]
    RM = R
    for i in range(P):
        RM[3 * i, i] = 1

    return RM


def relu(a):
    """Implements ReLU activation function"""
    return np.maximum(0, a)


def relu_back(a):
    """Implements ReLU derivative on a matrix"""

    a[a < 0] = 0
    a[a >= 0] = 1

    return a


def get_random_batch(batches_train, batch_size):
    """Outputs random batch of batch_size"""

    idx = np.random.randint(batches_train.shape[0], size=batch_size)

    return batches_train[idx, :]


def get_loss(Y_batch, X_batch_train):
    """Claculates sum squared loss"""

    loss = np.sum(np.power((Y_batch - X_batch_train), 2))

    return loss
