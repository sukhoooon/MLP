from keras. datasets import boston_housing
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. data generation - 30 feature, 2 class
whole_data = load_breast_cancer()

X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 7)

# 2. creating model
model = Sequential()
model.add(Dense(10, input_shape = (30,), activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

sgd = optimizers.SGD(lr = 0.01)
model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
# model.summary()

# 3. model fitting
model.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose = 1)

# 4. model evaluation
results = model.evaluate(X_test, y_test)
print('loss: ', results[0])
print('accuracy: ', results[1])