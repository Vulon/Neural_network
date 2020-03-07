from Network import Network
import activate
import numpy as np
import random
import math

def test_activations():
    print("Test sigmoid")
    x = [random.randint(1, 20) for i in range(10)]
    x = np.array(x).reshape((len(x), 1))
    def sigmoid(a):
        return 1 / (1 + math.e ** (-a))
    def sigmoid_derivative(a):
        return sigmoid(a) * (1 - sigmoid(a))
    sigmoid_y = sigmoid(x)
    assert np.min(sigmoid_y == activate.sigma(x)) == 1
    assert np.min(sigmoid_derivative(x) == activate.sigma_der(x)) == 1

def test_loss():
    print("Test MSE loss")
    x = [random.randint(1, 20) for i in range(10)]
    x_hat = np.array([entry + random.randint(-5, 5) for entry in x])
    x = np.array(x).reshape((len(x), 1))
    x_hat = x_hat.reshape((x_hat.size, 1))
    def MSE(a, b):
        return (a - b) ** 2 / 2
    def MSE_derivative(a, b):
        return b - a
    assert np.min(MSE(x, x_hat) == activate.mean_square_error(x, x_hat)) == 1
    assert np.min(MSE_derivative(x, x_hat) == activate.mean_square_error_der(x, x_hat)) == 1

def test_forward():
    print("Test net forwarding")
    def sigmoid(a):
        return 1 / (1 + math.e ** (-a))
    Net = Network()
    Net.add_layer(4, activate.SIGMA, input_size=3)
    layer = Net.layers[0]
    for count in range(5):
        x = np.random.rand(3, 1)
        product = sigmoid(np.dot(layer.weights, x) + layer.bias)
        product_hat = layer.forward(x)
        assert np.min(product == product_hat) == 1


def test_iris_dataset(epochs, batch_size, neurons_list, learn_speed=0.15, r2=0.1):
    print("Test Network on Iris dataset")
    from sklearn import datasets
    iris = datasets.load_iris()
    permutations = [x for x in range(len(iris.data))]
    random.shuffle(permutations)
    permutations = np.array(permutations)
    iris_data = np.array(iris.data)[permutations]
    iris_target = np.array(iris.target)[permutations]

    train_x = iris_data[: -20]
    train_y = iris_target[: -20]
    test_x = iris_data[-20:]
    test_y = iris_target[-20:]

    def one_hot(a, classes: int):
        arr = np.zeros((len(a), classes))

        for i in range(len(a)):
            arr[i, a[i]] = 1
        return arr

    train_y = one_hot(train_y, len(iris.target_names))
    test_y = one_hot(test_y, len(iris.target_names))

    Net = Network()
    Net.add_loss(activate.CROSS_ENTROPY)
    for i, n in enumerate(neurons_list):
        if i == 0:
            Net.add_layer(n, activate.SIGMA, input_size=4)
        else:
            Net.add_layer(n, activate.SIGMA)

    Net.set_regularization_constant(r2)
    Net.fit(train_x, train_y, epochs, batch_size, learn_speed=learn_speed)
    Net.show_weights()
    return Net.test(test_x, test_y)

test_activations()
test_loss()
test_forward()

loss, accuracy = np.array(test_iris_dataset(240, 10, [6, 3], learn_speed=0.4, r2=0.001))
print("Loss:", loss)
print("Accuracy:", accuracy)