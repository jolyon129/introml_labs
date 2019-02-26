# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

with open("reviewstrain.txt", "r") as file:
    X_train = []
    y_train = []
    for line in file:
        # exclude the label
        X_train.append(line[1:].strip())
        y_train.append(int(line[0]))

with open("reviewstest.txt", "r") as file:
    X_test = []
    y_test = []
    for line in file:
        X_test.append(line[1:].strip())
        y_train.append(int(line[0]))

corpus = X_train + X_test
# %%
vectorizer = CountVectorizer()
vectorizer.fit(corpus)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

# %%


class K_nearest:
    def __init__(self, k, dis_func=None):
        self.k = k
        self.dis_func = dis_func

    def fit(self, X_train):
        self.X_train = X_train
        return self

    def predict(self, X):
        y_hat = np.zeros(len(X)).reshape(-1, 1)
        for i1 in range(len(X)):
            distances = [None]*self.X_train.shape[0]
            x_set = set(X[i1].indices)
            for i2 in range(self.X_train.shape[0]):
                inter = x_set.intersection(self.X_train[i2].indices)
                distances[i2] = 1 / \
                    len(inter) if len(inter) != 0 else self.X_train.shape[1]+1
            sorted_idx = np.argsort(distances)
            flag = 0
            for i in range(5, len(sorted_idx)):
                if distances[sorted_idx[i]] != distances[sorted_idx[i-1]]:
                    break
            flag = i
            nearest_idx = sorted_idx[:flag]
            nearest_labels = np.asarray(y_train)[[*nearest_idx]]
            y_hat[i1] = 1 if nearest_labels.mean() >= 0.5 else 0
        return y_hat

# %%
test = []
test.append(X_test[17])
test.append(X_test[18])
k_n = K_nearest(1).fit(X_train)
y_hat = k_n.predict(test)
print(y_hat)
# %%
