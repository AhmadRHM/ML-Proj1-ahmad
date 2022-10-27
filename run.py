from helpers import *
import matplotlib.pyplot as plt
from implementations import mean_squared_error_gd, mean_squared_error_sgd, least_squares, ridge_regression, \
    logistic_regression, reg_logistic_regression, predict_simple, predict_logistic, penalized_logistic_regression
import warnings
warnings.filterwarnings("ignore")

do_balancing = False
do_poly = True
use_pca = True

y, x, ids = load_csv_data("train.csv", sub_sample=False)

one_hotted = np.zeros((x.shape[0], 3))
for i in range(3):
    one_hotted[x[:, 22] == i, i] = 1
x = np.delete(x, 22, 1)
x = np.concatenate([x, one_hotted], 1)

y[y == -1] = 0

def return_poly(x):
    a_copy = x.copy()

    # deg = 4
    # x = np.concatenate([a_copy] + [a_copy**i for i in range(2, deg)], 1)

    for i in range(a_copy.shape[1]):
        for j in range(i, a_copy.shape[1]):
            new_col = a_copy[:, i] * a_copy[:, j]
            new_col = np.reshape(new_col, (-1, x.shape[0]))
            x = np.concatenate([x, new_col.T], 1)
    x = np.concatenate([x, a_copy**3, a_copy**4, a_copy**5], 1)
    # x = np.concatenate([x, np.log(a_copy)], 1)
    # for i in range(a_copy.shape[1]):
    #     for j in range(i, a_copy.shape[1]):
    #         for k in range(j, a_copy.shape[1]):
    #             new_col = a_copy[:, i] * a_copy[:, j] * a_copy[:, k]
    #             new_col = np.reshape(new_col, (-1, x.shape[0]))
    #             x = np.concatenate([x, new_col.T], 1)

    return x

x_train, y_train = x[:200000], y[:200000]


def plot_balance(y):
    counts = np.unique(y, return_counts=True)
    plt.bar(counts[0], counts[1], tick_label=['class 0', 'class 1'])
    plt.title('count of classes')
    plt.xlabel('class')
    plt.ylabel('count')
    plt.show()

# plot_balance(y)

def balance(x_train, y_train):
    class1_num, class2_num = (y_train == 0).sum(), (y_train == 1).sum()
    if class1_num < class2_num:
        clone_ids = np.random.choice(np.where(y_train == 0)[0], class2_num - class1_num)
        x_train = np.concatenate([x_train, x_train[clone_ids]], 0)
        y_train = np.concatenate([y_train, np.zeros(class2_num - class1_num)])
    elif class1_num > class2_num:
        clone_ids = np.random.choice(np.where(y_train == 1)[0], class1_num - class2_num)
        x_train = np.concatenate([x_train, x_train[clone_ids]], 0)
        y_train = np.concatenate([y_train, np.ones(class1_num - class2_num)])
    return x_train, y_train


# do balancing and a random shuffle
if do_balancing:
    x_train, y_train = balance(x_train, y_train)
    random_ids = np.arange(len(x_train))
    np.random.shuffle(random_ids)
    x_train = x_train[random_ids]
    y_train = y_train[random_ids]
    plot_balance(y_train)

# Plotting missing percentages by columns
miss_perc = [np.sum(x_train[:, i] == -999) / x_train.shape[0] for i in range(x_train.shape[1])]
# plt.bar(list(range(x_train.shape[1])), miss_perc)
# plt.show()

# plt.hist(x_train[:, 0][x_train[:, 0] != -999])
# plt.show()


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

    x_transformed = x_std.dot(eig_vectors[:, idx[:n]])

    return x_transformed, eig_vectors[:, idx[:n]], eig_values


def preprocess(x_train, x_validation, miss_threshold=0.4):
    # remove missing columns
    miss_perc = [np.sum(x[:, i] == -999) / x.shape[0] for i in range(x.shape[1])]
    removing_columns = np.array(miss_perc) > miss_threshold
    x_train = x_train[:, ~removing_columns]
    x_validation = x_validation[:, ~removing_columns]

    # fill missing values for column 0
    x_train, x_validation = fill_missing(x_train, x_validation, 0, np.median)

    if use_pca:
        x_train_transformed, eig_vectors_train, x_train_eig_values = pca_transform(x_train, n=8)
        # for i in range(8):
        #     plt.hist(x_train_transformed[:, i], bins=20)
        #     plt.show()
        mean = np.mean(x_train, axis=0)
        var = np.var(x_train - mean, axis=0)
        x_validation_transformed = ((x_validation - mean) / var) @ eig_vectors_train
    else:
        x_train_transformed, x_validation_transformed, x_train_eig_values = x_train, x_validation, None

    if do_poly:
        x_train_transformed = return_poly(x_train_transformed)
        x_validation_transformed = return_poly(x_validation_transformed)

        print("poly done! :D", x_train_transformed.shape[1])

    return x_train_transformed, x_validation_transformed, x_train_eig_values


def plot_ceverage(coverage, num_feature):
    plt.plot(coverage[:num_feature])
    plt.grid()
    plt.yticks(np.arange(0, 105, 5))
    plt.xlabel('number of features')
    plt.ylabel('coverage')
    plt.title('Cumulative Coverage Percent of Variance')
    plt.show()


cross_validation_losses, cross_validation_accuracy = [], []
for validation_partition in range(1):
    begin, end = validation_partition * 50_000, (validation_partition + 1) * 50_000
    x_train, y_train = np.concatenate([x[:begin], x[end:]], 0), np.concatenate([y[:begin], y[end:]], 0)
    x_validation, y_validation = x[begin:end], y[begin:end]

    # do balancing and a random shuffle
    if do_balancing:
        x_train, y_train = balance(x_train.copy(), y_train.copy())

        random_ids = np.arange(len(x_train))
        np.random.shuffle(random_ids)
        x_train = x_train[random_ids]
        y_train = y_train[random_ids]

    # do preprocess
    x_train_transformed, x_validation_transformed, x_train_eig_values = preprocess(x_train, x_validation)
    x_train_transformed = np.concatenate([np.ones((len(x_train_transformed), 1)), x_train_transformed], 1)
    x_validation_transformed = np.concatenate([np.ones((len(x_validation_transformed), 1)), x_validation_transformed], 1)

    # do training and validation
    # w, loss = mean_squared_error_gd(y_train, x_train_transformed, np.random.random(x_train_transformed.shape[1]), 10000, 0.001)
    training_iter, validation_iter, training_loss, validation_loss, training_accuracy, validation_accuracy = [], [], [], [], [], []
    w = np.random.random(x_train_transformed.shape[1])
    lr = 0.001
    for step in range(1, 2001):
        w, loss = reg_logistic_regression(y_train, x_train_transformed, 0, w, 10, lr)
        # lr /= 1.0002
        training_iter.append(step * 10)
        training_loss.append(loss)
        training_accuracy.append((predict_logistic(x_train_transformed, w) == y_train).sum() / len(y_train))

        if step % 100 == 0:
            validation_iter.append(step * 10)
            validation_loss.append(penalized_logistic_regression(y_validation, x_validation_transformed, w, 0.1)[0])
            validation_accuracy.append((predict_logistic(x_validation_transformed, w) == y_validation).sum() / len(y_validation))
            print(step, validation_accuracy[-1], lr)

    plt.plot(training_iter, training_loss, label="train")
    plt.plot(validation_iter, validation_loss, label="validation")
    plt.grid()
    plt.xlabel('num iter')
    plt.ylabel('loss')
    plt.title('Loss curve')
    plt.legend()
    plt.show()

    plt.plot(training_iter, training_accuracy, label="train")
    plt.plot(validation_iter, validation_accuracy, label="validation")
    plt.grid()
    plt.xlabel('num iter')
    plt.ylabel('accuracy')
    plt.title('accuracy curve')
    plt.legend()
    plt.show()

    cross_validation_losses.append(loss)
    cross_validation_accuracy.append((predict_logistic(x_validation_transformed, w) == y_validation).sum() / len(y_validation))
    # if validation_partition == 0:
    #     coverage_train = np.cumsum(x_train_eig_values) / np.sum(x_train_eig_values) * 100
    #     plot_ceverage(coverage_train, 10)

