import numpy as np
class K_nearest:
    def __init__(self, k, dis_func=None):
        self.k = k
        self.dis_func = dis_func

    def fit(self, X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X):
        y_hat = np.zeros(X.shape[0]).reshape(-1, 1)
        for i1 in range(X.shape[0]):
            distances = [None]*self.X_train.shape[0]
            x_set = set(X[i1].indices)
            for i2 in range(self.X_train.shape[0]):
                inter = x_set.intersection(self.X_train[i2].indices)
                distances[i2] = 1 / \
                    len(inter) if len(inter) != 0 else self.X_train.shape[1]+1
            sorted_idx = np.argsort(distances)
            flag = 0
            for i in range(self.k, len(sorted_idx)):
                if distances[sorted_idx[i]] != distances[sorted_idx[i-1]]:
                    break
            flag = i
            nearest_idx = sorted_idx[:flag]
            nearest_labels = np.asarray(self.y_train)[[*nearest_idx]]
            y_hat[i1] = 1 if nearest_labels.mean() >= 0.5 else 0
        return y_hat
