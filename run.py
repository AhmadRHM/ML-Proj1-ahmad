from helpers import *
import matplotlib.pyplot as plt
from implementations import mean_squared_error_gd, mean_squared_error_sgd, least_squares, ridge_regression, \
    logistic_regression, reg_logistic_regression
import warnings
warnings.filterwarnings("ignore")


y, x, ids = load_csv_data("train.csv", sub_sample=False)


x_train, y_train = x[:200000], y[:200000]


def plot_balance(y):
    counts = np.unique(y, return_counts=True)
    plt.bar(counts[0], counts[1], tick_label=['class 0', 'class 1'])
    plt.title('count of classes')
    plt.xlabel('class')
    plt.ylabel('count')
    plt.show()

# Plotting missing percentages by columns
miss_perc = [np.sum(x_train[:, i] == -999) / x_train.shape[0] for i in range(x_train.shape[1])]
plt.bar(list(range(x_train.shape[1])), miss_perc)
plt.show()

plt.hist(x_train[:, 0][x_train[:, 0] != -999])
plt.show()


def find_mod(a):
    values, counts = np.unique(a, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]


def fill_missing(x_train, x_validation, col_idx, func):
    replace_val = func(x_train[:, col_idx])

    x_train[:, col_idx] = np.where(x_train[:, col_idx] != -999, x_train[:, col_idx], replace_val)
    x_validation[:, col_idx] = np.where(x_validation[:, col_idx] != -999, x_validation[:, col_idx], replace_val)
    return x_train, x_validation


def normalize(x):
    x_out = x.copy()
    x_out = x_out - np.mean(x_out, axis=0)
    x_out = x_out / np.var(x_out, axis=0)
    return x_out


def pca_transform(x_raw, n=5):
    x_std = normalize(x_raw)
    cov = np.cov(x_std.T)
    eig_values, eig_vectors = np.linalg.eig(cov)

    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]

    x_transformed = x_raw.dot(eig_vectors[:, idx[:n]])

    return x_transformed, eig_vectors[:, idx[:n]], eig_values


def preprocess(x_train, x_validation, miss_threshold=0.4):
    # remove missing columns
    miss_perc = [np.sum(x[:, i] == -999) / x.shape[0] for i in range(x.shape[1])]
    removing_columns = np.array(miss_perc) > miss_threshold
    x_train = x_train[:, ~removing_columns]
    x_validation = x_validation[:, ~removing_columns]

    # fill missing values for column 0
    x_train, x_validation = fill_missing(x_train, x_validation, 0, find_mod)

    x_train_transformed, eig_vectors_train, x_train_eig_values = pca_transform(x_train, n=5)
    x_validation_transformed = x_validation @ eig_vectors_train

    return x_train_transformed, x_validation_transformed, x_train_eig_values


def plot_ceverage(coverage, num_feature):
    plt.plot(coverage[:num_feature])
    plt.grid()
    plt.yticks(np.arange(0, 105, 5))
    plt.xlabel('number of features')
    plt.ylabel('coverage')
    plt.title('Cumulative Coverage Percent of Variance')
    plt.show()


validation_losses, validation_accuracy = [], []
for validation_partition in range(5):
    begin, end = validation_partition * 50_000, (validation_partition + 1) * 50_000
    x_train, y_train = np.concatenate([x[:begin], x[end:]], 0), np.concatenate([y[:begin], y[end:]], 0)
    x_validation, y_validation = x[begin:end], y[begin:end]

    x_train_transformed, x_validation_transformed, x_train_eig_values = preprocess(x_train, x_validation)
    x_train_transformed = np.concatenate([np.ones((len(x_train_transformed), 1)), x_train_transformed], 1)

    # w, loss = mean_squared_error_gd(y_train, x_train_transformed, np.random.random(x_train_transformed.shape[1]), 10000, 0.001)
    w, loss = reg_logistic_regression(y_train, x_train_transformed, 0.1, np.random.random(x_train_transformed.shape[1]), 10000, 0.001)
    validation_losses.append(loss)
    if validation_partition == 0:
        coverage_train = np.cumsum(x_train_eig_values) / np.sum(x_train_eig_values) * 100
        plot_ceverage(coverage_train, 10)

