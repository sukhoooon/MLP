import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import numpy as np

# "https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/1.%20MLP/2-Advanced-MLP/.ipynb_checkpoints/2-Advanced-MLP-checkpoint.ipynb"
from tensorflow_core import optimizers

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plotting raw mnist data
# plt.imshow(X_train[0])    # show first number in the dataset
# plt.show()
# print('Label: ', y_train[0])

# reshaping X data: (n, 28, 28) => (n, 784)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

# converting y data into categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# dropping data(using 33% of raw data)
X_train, _ , y_train, _ = train_test_split(X_train, y_train, test_size = 0.67, random_state = 7)

# 1. weight initialization
def mlp_model():
    model = Sequential()

    model.add(Dense(50, input_shape=(784,), kernel_initializer='he_normal'))  # use he_normal initializer
    model.add(Activation('sigmoid'))
    model.add(Dense(50, kernel_initializer='he_normal'))  # use he_normal initializer
    model.add(Activation('sigmoid'))
    model.add(Dense(50, kernel_initializer='he_normal'))  # use he_normal initializer
    model.add(Activation('sigmoid'))
    model.add(Dense(50, kernel_initializer='he_normal'))  # use he_normal initializer
    model.add(Activation('sigmoid'))
    model.add(Dense(10, kernel_initializer='he_normal'))  # use he_normal initializer
    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
model = mlp_model()
history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)
results = model.evaluate(X_test, y_test)
print('Test accuracy: ', results[1])

# 2. relu/selu nonlinearity activation function
def mlp_model():
    model = Sequential()

    model.add(Dense(50, input_shape=(784,)))
    model.add(Activation('relu'))  # use relu
    model.add(Dense(50))
    model.add(Activation('relu'))  # use relu
    model.add(Dense(50))
    model.add(Activation('relu'))  # use relu
    model.add(Dense(50))
    model.add(Activation('relu'))  # use relu
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
model = mlp_model()
history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)
results = model.evaluate(X_test, y_test)
print('Test accuracy: ', results[1])

# 3. optimizer
def mlp_model():
    model = Sequential()

    model.add(Dense(50, input_shape=(784,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr=0.001)  # use Adam optimizer
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
model = mlp_model()
history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)
results = model.evaluate(X_test, y_test)
print('Test accuracy: ', results[1])

# 4. batch normalization
from keras.layers import BatchNormalization


def mlp_model():
    model = Sequential()

    model.add(Dense(50, input_shape=(784,)))
    model.add(BatchNormalization())  # Add Batchnorm layer before Activation
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(BatchNormalization())  # Add Batchnorm layer before Activation
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(BatchNormalization())  # Add Batchnorm layer before Activation
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(BatchNormalization())  # Add Batchnorm layer before Activation
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = mlp_model()
history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)
results = model.evaluate(X_test, y_test)
print('Test accuracy: ', results[1])

# 5. dropout
from keras.layers import Dropout


def mlp_model():
    model = Sequential()

    model.add(Dense(50, input_shape=(784,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))  # Dropout layer after Activation
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))  # Dropout layer after Activation
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))  # Dropout layer after Activation
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))  # Dropout layer after Activation
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
model = mlp_model()
history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)

results = model.evaluate(X_test, y_test)
print('Test accuracy: ', results[1])

# 6. model emsemble
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis = 1)


def mlp_model():
    model = Sequential()

    model.add(Dense(50, input_shape=(784,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model1 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)
model2 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)
model3 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)


ensemble_clf = VotingClassifier(estimators = [('model1', model1), ('model2', model2), ('model3', model3)], voting = 'soft')
ensemble_clf.fit(X_train, y_train)
y_pred = ensemble_clf.predict(X_test)
print('Test accuracy:', accuracy_score(y_pred, y_test))