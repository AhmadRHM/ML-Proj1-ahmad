import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from implementations import mean_squared_error_gd, mean_squared_error_sgd, least_squares, ridge_regression, \
    logistic_regression, reg_logistic_regression, predict_simple, predict_logistic, penalized_logistic_regression, sigmoid

y, x, ids = load_csv_data("train.csv", sub_sample=False)

xs, ys = [], []
population = []
for i in range(4):
    xs.append(x[x[:, 22] == i])
    ys.append(y[x[:, 22] == i])
    population.append(xs[-1].shape[0] / x.shape[0])
population = np.array(population)

# do some data cleaning and feature expansion
for i, x in enumerate(xs):
    # fill first column
    x = fill_missing(x, col_idx=0, func=np.median)


    # remove other meaningless columns
    var = np.var(x, axis=0)
    removing_cols = np.where(var == 0)[0]
    xs[i] = np.delete(x, removing_cols, 1)

    # normalize
    xs[i] -= np.mean(xs[i], axis=0)
    xs[i] /= np.std(xs[i], axis=0)

    # add poly
    x_copy = xs[i].copy()
    for k in range(x_copy.shape[1]):
        for j in range(k, x_copy.shape[1]):
            new_col = x_copy[:, k] * x_copy[:, j]
            new_col = np.reshape(new_col, (x_copy.shape[0], -1))
            xs[i] = np.concatenate([xs[i], new_col], 1)
    xs[i] = np.concatenate([xs[i], x_copy ** 3, np.cos(x_copy), np.sin(x_copy), x_copy**4, x_copy**5, x_copy**6, sigmoid(x_copy), np.tanh(x_copy), np.sqrt(np.abs(x_copy)), np.sqrt(np.sqrt(np.abs(x_copy))), np.log(1 + np.abs(x_copy))], 1)
    if i > 2:
        xs[i] = np.concatenate([xs[i], np.sinh(x_copy)], 1)


validation_accuracy = []
ws = []
for i, x, y in zip(range(len(xs)), xs, ys):
    N = len(x)
    thresh = int(0.33*N)
    x_train, x_validation = x[:thresh], x[thresh:]
    y_train, y_validation = y[:thresh], y[thresh:]

    x_train, x_validation = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], 1), np.concatenate([x_validation, np.ones((x_validation.shape[0], 1))], 1)
    lambda_ = 0
    if i == 1:
        lambda_ = 0.0000002
    w, loss = ridge_regression(y_train, x_train, lambda_)
    ws.append(w)
    print("train accuracy:", (predict_simple(x_train, w) == y_train).mean())
    print("validation accuracy:", (predict_simple(x_validation, w) == y_validation).mean())
    validation_accuracy.append((predict_simple(x_validation, w) == y_validation).mean())

print((validation_accuracy * population).sum())

y, x, ids = load_csv_data("test.csv", sub_sample=False)

xs, idss = [], []
for i in range(4):
    xs.append(x[x[:, 22] == i])
    idss.append(ids[x[:, 22] == i])

# for x_train in xs:
#     miss_perc = [np.sum(x_train[:, i] == -999) / x_train.shape[0] for i in range(x_train.shape[1])]
#     plt.bar(list(range(x_train.shape[1])), miss_perc)
#     plt.show()

# do some data cleaning and feature expansion
labels = []
for i, x_test in enumerate(xs):
    print("i:", i)
    # fill first column
    x_test = fill_missing(x_test, col_idx=0, func=np.median)

    # remove other meaningless columns
    var = np.var(x_test, axis=0)
    removing_cols = np.where(var == 0)[0]
    x_test = np.delete(x_test, removing_cols, 1)

    # normalize
    x_test -= np.mean(x_test, axis=0)
    x_test /= np.std(x_test, axis=0)

    # add poly
    x_copy = x_test.copy()
    for k in range(x_copy.shape[1]):
        for j in range(k, x_copy.shape[1]):
            new_col = x_copy[:, k] * x_copy[:, j]
            new_col = np.reshape(new_col, (x_copy.shape[0], -1))
            x_test = np.concatenate([x_test, new_col], 1)
    x_test = np.concatenate([x_test, x_copy ** 3, np.cos(x_copy), np.sin(x_copy), x_copy**4, x_copy**5, x_copy**6, sigmoid(x_copy), np.tanh(x_copy), np.sqrt(np.abs(x_copy)), np.sqrt(np.sqrt(np.abs(x_copy))), np.log(1 + np.abs(x_copy))], 1)
    if i > 2:
        x_test = np.concatenate([x_test, np.sinh(x_copy)], 1)

    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], 1)

    # calculate labels
    label = predict_simple(x_test, ws[i])
    labels.append(label)

ids = np.concatenate(idss)
pred_labels = np.concatenate(labels)
pred_labels = pred_labels[np.argsort(ids)]
ids = ids[np.argsort(ids)]
create_csv_submission(ids, pred_labels, "submission_3.csv")





