import numpy as np
import pandas as pd

# Prapare Data

df = pd.read_csv('./spambasetrain.csv', header=None)
train_set = df
train_x = df[df.columns[:9]].values
train_y = df[df.columns[-1]].values
x_y_0 = (train_set.loc[train_set[9] == 0])[df.columns[:9]]
x_y_1 = (train_set.loc[train_set[9] == 1])[df.columns[:9]]

df_test = pd.read_csv('./spambasetest.csv', header=None)
test_x = df_test[df_test.columns[:9]]
test_x = test_x.values
test_y = df_test[df_test.columns[9]].values

# Calculate Means

means = np.zeros(shape=(2, 9))
means[0] = (np.mean(x_y_0, axis=0))
means[1] = (np.mean(x_y_1, axis=0))

x_y_0 = x_y_0.values
x_y_1 = x_y_1.values

# Calculate Variances

variances = np.zeros(shape=(2, 9))
variances[0] = (1. / (len(x_y_0) - 1.)) * np.sum((x_y_0 - means[0]) ** 2, axis=0)
variances[1] = (1. / (len(x_y_1) - 1.)) * np.sum((x_y_1 - means[1]) ** 2., axis=0)


def gaussian(x, mu, var):
    t = np.exp(-(x - mu) ** 2. / (2. * var)) * 1. / (np.sqrt(2 * np.pi * var))
    return t


# Calculate likelihood
likelihood = [None, None]
likelihood[0] = gaussian(test_x, means[0], variances[0])
likelihood[1] = gaussian(test_x, means[1], variances[1])
likelihood = np.asarray(likelihood)

# Calculate prior
prior = [None, None]
prior[0] = train_set.loc[train_set[9] == 0].shape[0] / train_set.shape[0]
prior[1] = train_set.loc[train_set[9] == 1].shape[0] / train_set.shape[0]
prior = np.asarray(prior)

# predict:
log_p = np.sum(np.log(likelihood), axis=2) + prior[:, None]
prediciton = (log_p[0] <= log_p[1]).astype(np.int)

labels = (prediciton == test_y)

pd.DataFrame(prior).to_csv('priors.csv',header=False)
pd.DataFrame(prediciton).to_csv('predictions.csv',header=False)
new_path = './results.txt'
results_txt = open(new_path, 'w')
results_txt.write(f'P(C=0) = {prior[0]}, P(C=1) = {prior[1]} \n')
str1 = ''
str2 = ''
for i in range(9):
    str1 += f'{means[0][i], variances[0][i]}, \n'
    str2 += f'{means[1][i], variances[1][i]}, \n'

results_txt.writelines('For C = 0, the mean and variance for each feature is as follow:\n')
results_txt.writelines(str1)
results_txt.writelines('For C = 1, the mean and variance for each feature is as follow:\n')
results_txt.writelines(str2)

results_txt.writelines(f'Correct prediction:{len([i for i in labels if i == True])}\n')
results_txt.writelines(f'Correct prediction:{len([i for i in labels if i != True])}\n')
results_txt.writelines(f'The percentage error:{len([i for i in labels if i != True])/len(labels)}\n')
results_txt.close()
