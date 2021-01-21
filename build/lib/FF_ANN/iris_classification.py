import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scaling import Standardization
from layers import Dense
from activation_function import ReLU
from network_training import train, predict, iterate_minibatches

np.random.seed(101) # For reproducibility

# Load iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create train and test sets
order = np.argsort(np.random.random(len(X)))
X = X[order]
y = y[order]

#125 for train, 25 for test
X_train = X[:125]
y_train = y[:125]

X_test = X[125:]
y_test = y[125:]


# Scaling
zsc = Standardization()
zsc.fit(X_train)
X_train = zsc.transform(X_train)
X_test = zsc.transform(X_test)

# Validation set (25)
X_train, X_val = X_train[:-25], X_train[-25:]
y_train, y_val = y_train[:-25], y_train[-25:]

# Build network architecture
network = []
network.append(Dense(X_train.shape[1], 10))
network.append(ReLU())
network.append(Dense(10, 20))
network.append(ReLU())
network.append(Dense(20, 3))

# Train the model
train_log = []
val_log = []

for epoch in range(50):

    for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=16):
        train(network, x_batch, y_batch)

    train_log.append(np.mean(predict(network, X_train) == y_train))
    val_log.append(np.mean(predict(network, X_val) == y_val))

    print("Epoch", epoch)
    print("Train accuracy:", train_log[-1])
    print("Val accuracy:", val_log[-1])


# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(train_log, label='train accuracy')
plt.plot(val_log, label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()


# prediction on the test set
print("Test accuracy", np.mean(predict(network, X_test) == y_test))