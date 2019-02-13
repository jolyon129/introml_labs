import numpy as np
import pandas as pd

df = pd.read_csv('./spambasetrain.csv')
df_test = pd.read_csv('./spambasetest.csv', header=None)
test_x = df_test[df_test.columns[:9]]
train_set = df
train_x = df[df.columns[:9]].values
train_y = df[df.columns[-1]].values
x_y_0 = (train_set.loc[train_set[9] == 0])[df.columns[:9]].values
x_y_1 = (train_set.loc[train_set[9] == 1])[df.columns[:9]].values


def gaussian(x, mu, var):
    t = np.exp([-(x - mu) ** 2 / (2 * var)]) * 1 / (np.sqrt(2 * np.pi * var))
    return t.squeeze()


def calculate_prior():
    prior = [None, None]
    prior[0] = train_set.loc[train_set[9] == 0].shape[0] / train_set.shape[0]
    prior[1] = train_set.loc[train_set[9] == 1].shape[0] / train_set.shape[0]
    prior = np.asarray(prior)
    return prior


def calculate_means(x_y_0, x_y_1):
    means = np.zeros(shape=(2, 9))
    means[0] = (np.mean(x_y_0, axis=0))
    means[1] = (np.mean(x_y_1, axis=0))
    return means


def calculate_variances(x_y_0, x_y_1, means):
    variances = np.zeros(shape=(2, 9))
    variances[0] = (1. / (len(x_y_0) - 1.)) * np.sum((x_y_0 - means[0]) ** 2, axis=0)
    variances[1] = (1. / (len(x_y_1) - 1.)) * np.sum((x_y_1 - means[1]) ** 2, axis=0)
    return variances


def calculate_likelyhood(test_x, means, variances):
    likelihood = [None, None]
    likelihood[0] = gaussian(test_x, means[0], variances[0])
    likelihood[1] = gaussian(test_x, means[1], variances[1])
    likelihood = np.asarray(likelihood)
    return likelihood


def predict(likelihood, prior):
    log_p = np.sum(np.log(likelihood), axis=2) + prior[:, None]
    prediciton = (log_p[0] <= log_p[1]).astype(np.int)

