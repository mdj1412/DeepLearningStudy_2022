import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

#one hot encoding
Y = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

#분류 클래스 수
nb_classes = 3

W = tf.Variable(tf.random.normal([4, nb_classes]), name = 'weight')
b = tf.Variable(tf.random.normal([nb_classes]), name = 'bias')

def H(X): #use softmax function
    return tf.nn.softmax(tf.matmul(tf.cast(X, tf.float32), W) + b)


#w1g_list = []
learning_rate = 1e-1
with tf.GradientTape(persistent = True) as tape:
    for i in range(0, 3001):
        Hypothesis = H(X)
        #reduce_sum(, axis = 1): 열의 방향으로 더하면서 2차원에서 1차원으로 축소
        #axis 지정 안 할 경우, 차원을 0으로 축소
        cost = tf.reduce_mean(-tf.reduce_sum(Y * -tf.math.log(Hypothesis), axis = 1))

        W_grad, b_grad = tape.gradient(cost, [W, b])
        W.assign_add(learning_rate * W_grad)
        b.assign_add(learning_rate * b_grad)
        #w1g_list.append(W_grad.numpy()[0][0])

        if i % 100 == 0:
            print("#", i, " cost: ", cost.numpy())
            #print("W_grad: ", W_grad.numpy())
            #print("W: ", W.numpy())

#plt.scatter(range(0, 3001), w1g_list)
#plt.show()

X_test = [[2, 1, 3, 2]]
predict = H(X_test)
print("predict: ", predict.numpy(), tf.math.argmax(predict, axis = 1))