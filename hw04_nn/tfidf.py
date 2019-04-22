#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from K_nearest import K_nearest
from sklearn.metrics import confusion_matrix

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
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
X_train_trans = vectorizer.transform(X_train)
X_test_trans = vectorizer.transform(X_test)
y_train = np.asarray(y_train)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
#%%
k_nn_1 = K_nearest(1).fit(X_train_trans, y_train)
y_test_hat = k_nn_1.predict_tfidf(X_test_trans)
cf_matrix = confusion_matrix(y_test, y_test_hat)
acc = (cf_matrix[0, 0]+cf_matrix[1, 1])/cf_matrix.sum()
true_positive = cf_matrix[1, 1]/cf_matrix.sum()
false_positive = cf_matrix[0, 1]/cf_matrix.sum()
print('The confusion matrix:')
print(cf_matrix)
print(f'The accuracy is: {acc}.')
print(f'The true positive rate is: {true_positive}.')
print(f'The false positive is: {false_positive}.')
#%%
k_nn_5 = K_nearest(5).fit(X_train_trans, y_train)
y_test_hat = k_nn_5.predict_tfidf(X_test_trans)
cf_matrix = confusion_matrix(y_test, y_test_hat)
acc = (cf_matrix[0, 0]+cf_matrix[1, 1])/cf_matrix.sum()
true_positive = cf_matrix[1, 1]/cf_matrix.sum()
false_positive = cf_matrix[0, 1]/cf_matrix.sum()
print('The confusion matrix:')
print(cf_matrix)
print(f'The accuracy is: {acc}.')
print(f'The true positive rate is: {true_positive}.')
print(f'The false positive is: {false_positive}.')


#%%
