# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from K_nearest import K_nearest

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
        y_test.append(int(line[0]))

corpus = X_train + X_test
# %%
# Tokenize
vectorizer = CountVectorizer()
vectorizer.fit(corpus)
X_train_trans = vectorizer.transform(X_train)
X_test_trans = vectorizer.transform(X_test)
y_train = np.asarray(y_train)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# %%
'''
1.a.i)
'''
print(X_test[17])
k_n = K_nearest(1).fit(X_train_trans, y_train)
y_hat = k_n.predict(X_test_trans[17])
print(y_hat)
# %%
y_test_hat = k_n.predict(X_test_trans)
# %%
'''
1.a.ii)
'''
print('---------------------/n')
print('1-nearest classifier')
y_test = np.asarray(y_test).reshape(-1,1)
cf_matrix = confusion_matrix(y_test,y_test_hat)
print('The confusion matrix is ')
print(cf_matrix)
acc = (cf_matrix[0,0]+cf_matrix[1,1])/cf_matrix.sum()
true_positive = cf_matrix[1,1]/cf_matrix.sum()
false_positive = cf_matrix[0, 1]/cf_matrix.sum()
print(f'The accuracy is: {acc}.')
print(f'The true positive rate is: {true_positive}.')
print(f'The false positive is: {false_positive}.')

#%%
k_n_5 = K_nearest(5).fit(X_train_trans, y_train)
print(X_test[17])
y_hat = k_n_5.predict(X_test_trans[17])
print(f"The predicted label is: {y_hat}")
#%%
y_test_hat_2 = k_n_5.predict(X_test_trans)
#%%
print('---------------------/n')
print('5-nearest classifier')
y_test_hat_2 = np.asarray(y_test_hat_2).reshape(-1, 1)
cf_matrix = confusion_matrix(y_test, y_test_hat_2)
print('The confusion matrix is ')
print(cf_matrix)
acc = (cf_matrix[0, 0]+cf_matrix[1, 1])/cf_matrix.sum()
true_positive = cf_matrix[1, 1]/cf_matrix.sum()
false_positive = cf_matrix[0, 1]/cf_matrix.sum()
print(f'The accuracy is: {acc}.')
print(f'The true positive rate is: {true_positive}.')
print(f'The false positive is: {false_positive}.')
#%%

print(y_train.mean())
y_test = np.asarray(y_test)
y_hat_zero_r = np.ones_like(y_test)
cfm = confusion_matrix(y_test, y_hat_zero_r)
print('If we use the Zero-R classifier, the confusion matrix is:')
print(cfm)
#%%
kf = KFold(n_splits=5, random_state=42, shuffle=False)
k_num = [3, 7, 99]
accuracy = [None]*3
for k in range(len(k_num)):
    a = []
    for train_idx, validation_idx in kf.split(X_train_trans):
        # print(train_idx)
        X_train_fold = X_train_trans[[*train_idx]]
        y_train_fold = y_train[[*train_idx]]
        X_val_fold = X_train_trans[[*validation_idx]]
        y_val_fold = y_train[[*validation_idx]]
        k_nn_c = K_nearest(k).fit(X_train_fold, y_train_fold)
        y_val_fold_hat = k_nn_c.predict(X_val_fold)
        cfm = confusion_matrix(y_val_fold, y_val_fold_hat)
        acc = (cfm[0, 0]+cfm[1, 1])/cfm.sum()
        a.append(acc)
    accuracy[k] = (np.sum(a)/kf.n_splits)
print(f'The accuracy is listed: ')
print(accuracy[1:])
print('The best K should be 3')
#%%
k_nn_7 = K_nearest(7).fit(X_train_trans, y_train)
y_test_hat = k_nn_7.predict(X_test_trans)
cfm = confusion_matrix(y_test, y_test_hat)
acc = (cfm[0, 0]+cfm[1, 1])/cfm.sum()
print('The confusion matrix is ')
print(cfm)
print(f'The accuracy is: {acc}.')


