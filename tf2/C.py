import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

x_pred = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

#one hot encoding
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (4,)))
model.add(tf.keras.layers.Dense(units = 3, activation = 'softmax'))
sgd = tf.optimizers.SGD(learning_rate = 1e-2)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd)

history = model.fit(x_data, y_data, epochs = 300, batch_size = 1, shuffle = False)

y_pred = model.predict(x_pred)
print(tf.math.argmax(y_pred, axis = 1))