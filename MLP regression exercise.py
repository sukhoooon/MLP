from keras. datasets import boston_housing
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers

# 1. data generation
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# 2. model construction
model = Sequential()
model.add(Dense(10, input_shape=(X_train.shape[1], )))
model.add(Activation('sigmoid'))
model.add(Dense(10,))
model.add(Activation('sigmoid'))
model.add(Dense(10,))
model.add(Activation('sigmoid'))
model.add(Dense(1))

# model.add(Dense(10, input_shape=(X_train.shape[1], ), activation='sigmoid'))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(1))

sgd = optimizers.SGD(lr = 0.01)
model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mse'])
# model.summary()

# 3. model fitting
model.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose = 1)

# 4. model evaluation
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ', results[0])
print('mse : ', results[1])