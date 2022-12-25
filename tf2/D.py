import tensorflow as tf
import numpy as np

class Model:

    def __init__(self, name):
        self.name = name

    def build_model(self, x_train, y_train, x_test, y_test):
        tf.model = tf.keras.Sequential()

        tf.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), input_shape=(28,28,1), activation='relu', 
        kernel_initializer='he_normal'))
        tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        tf.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',
        kernel_initializer='he_normal'))
    
        tf.model.add(tf.keras.layers.Flatten())
        tf.model.add(tf.keras.layers.Dense(units=10, kernel_initializer='glorot_normal', activation='softmax'))

        tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
        tf.model.summary()
        
        tf.model.fit(x_train, y_train, epochs=30, batch_size=200)
        
    def predict(self, x_test):
        return tf.model.predict(x_test)

    def get_accuracy(self, x_test ,y_test):
        _, accuracy = tf.model.evaluate(x_test, y_test)
        return accuracy

models = []
num_models = 5
for m in range(num_models):
    models.append(Model("model" + str(m)))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

for m in models:
    m.build_model(x_train, y_train, x_test, y_test)

predictions = np.zeros([y_test.shape[0], y_test.shape[1]])
for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(x_test, y_test))
    p = m.predict(x_test)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_test, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', ensemble_accuracy)