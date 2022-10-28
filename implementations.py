import numpy as np
from matplotlib import pyplot as plt


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx.dot(w)
    N = y.shape[0]
    g_loss = -1 / N * tx.T @ e
    return g_loss


def compute_loss(y, tx, w):

    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    loss = 0.5 * (e ** 2).mean()
    return loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    return compute_gradient(y, tx, w)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    sigmoid(np.array([0.1]))
    array([0.52497919])
    sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    return 1/(1+np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    y = np.c_[[0., 1.]]
    tx = np.arange(4).reshape(2, 2)
    w = np.c_[[2., 3.]]
    round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    xw = tx @ w
    elementwise_loss = np.log(1+np.exp(xw)) - y*xw
    return elementwise_loss.mean()


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    np.set_printoptions(8)
    y = np.c_[[0., 1.]]
    tx = np.arange(6).reshape(2, 3)
    w = np.array([[0.1], [0.2], [0.3]])
    calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    # ***************************************************
    xw = tx @ w
    obser_n = y.shape[0]
    return 1 / obser_n * (tx.T @ (sigmoid(xw) - y))
    # ***************************************************


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    y = np.c_[[0., 1.]]
    tx = np.arange(6).reshape(2, 3)
    w = np.array([[0.1], [0.2], [0.3]])
    gamma = 0.1
    loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    round(loss, 8)
    0.62137268
    w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    grad = calculate_gradient(y, tx, w)
    updated_w = w - gamma*grad
    loss = calculate_loss(y, tx, w)
    return loss, updated_w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    y = np.c_[[0., 1.]]
    tx = np.arange(6).reshape(2, 3)
    w = np.array([[0.1], [0.2], [0.3]])
    lambda_ = 0.1
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    round(loss, 8)
    0.63537268
    gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    pen_loss = calculate_loss(y, tx, w)
    pen_grad = calculate_gradient(y, tx, w) + 2 * lambda_*w
    return pen_loss, pen_grad


# ======================================================================================


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y, tx, initial_w)]
    w = initial_w
    for n_iter in range(max_iters):
        g_loss = compute_gradient(y, tx, w)
        w = w - gamma * g_loss
        loss = compute_loss(y, tx, w)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    batch_size = 1
    ws = [initial_w]
    losses = [compute_loss(y, tx, initial_w)]
    w = initial_w

    N = y.shape[0]

    for n_iter in range(max_iters):
        for y_, tx_ in batch_iter(y, tx, batch_size):
            g_loss = compute_gradient(y_, tx_, w)
            w = w - gamma * g_loss
            loss = compute_loss(y_, tx_, w)
            ws.append(w)
            losses.append(loss)

            print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return ws[-1], losses[-1]


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """

    w_star = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    e = y - tx.dot(w_star)
    loss_star = 0.5 * (e ** 2).mean()
    return w_star, loss_star


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N = tx.shape[0]
    w_star = np.linalg.solve(tx.T.dot(tx) + lambda_ * 2 * N * (np.eye(tx.shape[1])), tx.T.dot(y))
    loss_star = np.linalg.norm((y - tx @ w_star), 2) ** 2 / (2 * N)
    return w_star, loss_star


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """just to get rid of the fucking error"""
    w = initial_w
    for n_iter in range(max_iters):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    loss = calculate_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """just to get rid of the fucking error"""
    w = initial_w
    for n_iter in range(max_iters):
        loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * grad
    loss = calculate_loss(y, tx, w)
    return w, loss

# ================================================================================


def predict_simple(tx, w):
    scores = tx @ w
    # dist_to_negative_one = np.abs((scores + 1) ** 2)
    # dist_to_one = np.sqrt((scores - 1) ** 2)
    labels = np.ones_like(scores) * -1
    # labels[dist_to_one < dist_to_negative_one] = 1
    labels[scores > 0] = 1
    return labels


def predict_logistic(tx, w):
    scores = sigmoid(tx @ w)
    labels = np.zeros_like(scores)
    labels[scores > 0.5] = 1
    return labels



