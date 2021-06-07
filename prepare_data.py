import numpy as np

from aux_code.ops import randomly_split_data
from aux_code.ops import one_hot_sequence


def load_data(dataset_name, seq_len=200):
    '''
    Returns:
    x - a n_samples long list containing arrays of shape (sequence_length,
                                                          n_features)
    y - an array of the labels with shape (n_samples, n_classes)
    '''
    print("Loading " + dataset_name + " dataset ...")

    if dataset_name == 'test':
        n_data_points = 5000
        sequence_length = 100
        n_features = 1
        x = list(np.random.rand(n_data_points, sequence_length, n_features))
        n_classes = 4
        y = np.random.randint(low=0, high=n_classes, size=n_data_points)

    if dataset_name == 'mnist':
        return get_mnist(permute=False)

    if dataset_name == 'pmnist':
        return get_mnist(permute=True)

    if dataset_name == 'add':
        x, y = get_add(n_data=150000, seq_len=seq_len)

    if dataset_name == 'copy':
        return get_copy(n_data=150000, seq_len=seq_len)

    train_idx, valid_idx, test_idx = randomly_split_data(
        y, test_frac=0.2, valid_frac=0.1)

    x_train = [x[i] for i in train_idx]
    y_train = y[train_idx]
    x_valid = [x[i] for i in valid_idx]
    y_valid = y[valid_idx]
    x_test = [x[i] for i in test_idx]
    y_test = y[test_idx]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_add(n_data, seq_len):
    x = np.zeros((n_data, seq_len, 2))
    x[:,:,0] = np.random.uniform(low=0., high=1., size=(n_data, seq_len))
    inds = np.random.randint(seq_len/2, size=(n_data, 2))
    inds[:,1] += seq_len//2
    for i in range(n_data):
        x[i,inds[i,0],1] = 1.0
        x[i,inds[i,1],1] = 1.0

    y = (x[:,:,0] * x[:,:,1]).sum(axis=1)
    y = np.reshape(y, (n_data, 1))
    return x, y


def get_copy(n_data, seq_len):
    x = np.zeros((n_data, seq_len+1+2*10))
    info = np.random.randint(1, high=9, size=(n_data, 10))

    x[:,:10] = info
    x[:,seq_len+10] = 9*np.ones(n_data)

    y = np.zeros_like(x)
    y[:,-10:] = info

    x = one_hot_sequence(x)
    y = one_hot_sequence(y)

    n_train, n_valid, n_test = [100000, 10000, 40000]
    x_train = list(x[:n_train])
    y_train = y[:n_train]
    x_valid = list(x[n_train:n_train+n_valid])
    y_valid = y[n_train:n_train+n_valid]
    x_test = list(x[-n_test:])
    y_test = y[-n_test:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_mnist(permute=False):
    import tensorflow.keras as keras

    (x_train, l_train), (x_test, l_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert l_train.shape == (60000,)
    assert l_test.shape == (10000,)
    NCLASSES = 10
    assert l_test.max() < NCLASSES
    assert l_train.max() < NCLASSES
    
    x_valid = x_train[-10000:]
    l_valid = l_train[-10000:]
    x_train = x_train[:-10000]
    l_train = l_train[:-10000]

    if permute:
        perm_mask = np.load('misc/pmnist_permutation_mask.npy')
    else:
        perm_mask = np.arange(784)

    x_train = list(np.array(x_train).reshape(50000, 784, 1)[:, perm_mask])
    x_valid = list(np.array(x_valid).reshape(10000, 784, 1)[:, perm_mask])
    x_test = list(np.array(x_test).reshape(10000, 784, 1)[:, perm_mask])
    y_train = keras.utils.to_categorical(l_train, NCLASSES)
    y_test = keras.utils.to_categorical(l_test, NCLASSES)
    y_valid = keras.utils.to_categorical(l_valid, NCLASSES)

    print("Train:Validation:Testing - %d:%d:%d" % (len(y_train), len(y_valid),
                                                   len(y_test)))

    return x_train, y_train, x_valid, y_valid, x_test, y_test
