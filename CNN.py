import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

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
y_train, y_cv = train.iloc[:, 0 ], cv.iloc[:, 0 ]
X_test = pd.read_csv('test.csv')/255

X_train = np.array(X_train).reshape((-1, 28, 28, 1))
X_cv = np.array(X_cv).reshape((-1, 28, 28, 1))
X_test = np.array(X_test).reshape((-1, 28, 28, 1))

onehot = OneHotEncoder()
onehot.fit(y_train.reshape(-1, 1))
y_train = onehot.transform(y_train.reshape(-1, 1)).toarray()
y_cv = onehot.transform(y_cv.reshape(-1, 1)).toarray()

cnn_clf = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(16, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

cnn_clf.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

generator = ImageDataGenerator(zoom_range = 0.1, 
                                height_shift_range = 0.1,
                                width_shift_range = 0.1,
                                rotation_range = 10)

# lr = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

cnn_clf.fit(X_train, y_train, epochs=5, batch_size=128, callbacks=[lr], 
            validation_data=(X_cv, y_cv))
cnn_clf.fit_generator(generator.flow(X_train, y_train, batch_size=16),
                        steps_per_epoch=1000,
                        epochs=15,
                        validation_data=(X_cv, y_cv),
                        callbacks=[lr])

final_loss, final_acc = cnn_clf.evaluate(X_cv, y_cv, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

y_pred = cnn_clf.predict(X_test)
y_pred = [np.argmax(i) for i in y_pred]
y_test_pred = pd.Series(y_pred, 
                        index=range(1, 28000+1), 
                        name='Label')
y_test_pred.to_csv('results.csv', index_label='ImageId', header=True)

