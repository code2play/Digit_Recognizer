import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
print()

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf


def view_digit(digit):
    digit = digit.reshape(28, 28)
    plt.imshow(digit, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()

data = pd.read_csv('train.csv')
# print(data.describe())
# view_digit(np.array(data.iloc[1, 1:]))

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, cv_index in split.split(data, data['label']):
    train = data.iloc[train_index, :]
    cv = data.iloc[cv_index, :]

X_train, X_cv = train.iloc[:, 1:]/255, cv.iloc[:, 1:]/255
# print(X_train.describe())
y_train, y_cv = train.iloc[:, 0 ], cv.iloc[:, 0 ]

X_test = pd.read_csv('test.csv')/255
# view_digit(np.array(X_test.iloc[0, :]))

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], 
                                        n_classes=10,
                                        feature_columns=feature_columns)
dnn_clf.fit(X_train, y_train, batch_size=64, steps=40000)

y_train_pred = list(dnn_clf.predict(X_train))
print('Training set accuracy: {}'.format(accuracy_score(y_train, y_train_pred)))

y_cv_pred = list(dnn_clf.predict(X_cv))
print('CV set accuracy: {}'.format(accuracy_score(y_cv, y_cv_pred)))

y_test_pred = pd.Series(list(dnn_clf.predict(X_test)), 
                        index=range(1, 28000+1), 
                        name='Label')
y_test_pred.to_csv('results.csv', index_label='ImageId', header=True)
