# %%
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
boston_data = load_boston()

'''
1.
'''
# %%


def predict_report(X_train, y_train, X_test, y_test):
    reg = LinearRegression().fit(X_train, y_train)
    y_train_hat = reg.predict(X_train)
    RSS_train = ((y_train_hat-y_train)**2).sum()
    TSS_train = ((y_train-y_train.mean())**2).sum()
    R_2_train = reg.score(X_train, y_train)
    print(
        f'RSS_train: {RSS_train}, TSS_train: {TSS_train}, R^2_train: {R_2_train}')
    y_test_hat = reg.predict(X_test)
    RSS_test = ((y_test_hat-y_test)**2).sum()
    TSS_test = ((y_test_hat-y_test.mean())**2).sum()
    R_2_test = reg.score(X_test, y_test)
    print(
        f'RSS_test: {RSS_test}, TSS_test: {TSS_test}, R^2_test: {R_2_test}')

    '''
        ridge regression
    '''

    MSEtr = []
    alphas = np.linspace(0, 1, 200)
    # alphas = [0,1e-3, 1e-2, 1e-1, 1]
    # Calculate the best alpha
    for a in alphas[1:]:
        reg_ridge = Ridge(alpha=a, solver='cholesky').fit(X_train, y_train)
        y_test_hat = reg_ridge.predict(X_test)
        MSE = np.mean((y_test_hat-y_test)**2)
        MSEtr.append(MSE)
    print(f'The best alpha is {alphas[np.argmin(MSEtr)+1]}')
    plt.plot(alphas[1:], MSEtr, 'o-')
    plt.xlim(0, 0.1)
    plt.grid()

    alphas[np.argmin(MSEtr)+1]
    reg_ridge = Ridge(alpha=alphas[np.argmin(MSEtr)+1],
                      solver='cholesky').fit(X_train, y_train)
    y_train_hat = reg_ridge.predict(X_train)
    RSS_train = ((y_train_hat-y_train)**2).sum()
    TSS_train = ((y_train-y_train.mean())**2).sum()
    R_2_train = reg_ridge.score(X_train, y_train)
    print(
        f'RSS_train: {RSS_train}, TSS_train: {TSS_train}, R^2_train: {R_2_train}')
    y_test_hat = reg_ridge.predict(X_test)
    RSS_test = ((y_test_hat-y_test)**2).sum()
    TSS_test = ((y_test_hat-y_test.mean())**2).sum()
    R_2_test = reg_ridge.score(X_test, y_test)
    print(
        f'RSS_test: {RSS_test}, TSS_test: {TSS_test}, R^2_test: {R_2_test}')

    return {reg, reg_ridge}

# %%
X = boston_data.data
y = boston_data.target.reshape(X.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
lin, lin_ridge =predict_report(X_train, y_train, X_test, y_test)

# %%
'''
2.
'''
poly_tranform = PolynomialFeatures(degree=2)
X_trans = poly_tranform.fit_transform(X)
X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(
    X_trans, y, test_size=0.2, random_state=42)
poly, poly_ridge = predict_report(
    X_train_trans, y_train_trans, X_test_trans, y_test_trans)
#%%
lin.coef_.shape
#%%
n = [[5, 0.5, 2, 0, 4, 8, 4, 6, 2, 2, 2, 4, 5.5]]
n_trans = poly_tranform.fit_transform(n)
price = poly_ridge.predict(n_trans)
print(f'The estimated value is : {price}')



