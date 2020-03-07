import Network
import numpy as np
import random
import activate


def test_iris_MSE(epochs, batch_size, neurons_list, learn_speed=0.15, r2=0.1):
    from sklearn import datasets
    iris = datasets.load_iris()
    permutations = [x for x in range(len(iris.data))]
    random.shuffle(permutations)
    permutations = np.array(permutations)
    iris_data = np.array(iris.data)[permutations]
    iris_target = np.array(iris.target)[permutations]


    train_x = iris_data[ : -20]
    train_y = iris_target[ : -20]
    test_x = iris_data[-20 : ]
    test_y = iris_target[-20 : ]
    
    def one_hot(a, classes : int):
        arr = np.zeros((len(a), classes))

        for i in range(len(a)):
            arr[i, a[i]] = 1
        return arr

    train_y = one_hot(train_y, len(iris.target_names))
    test_y = one_hot(test_y, len(iris.target_names))

    Net = Network.Network()
    Net.add_loss(activate.MSE)
    for i, n in enumerate(neurons_list):
        if i == 0 :
            Net.add_layer(n, activate.SIGMA, input_size=4)
        else:
            Net.add_layer(n, activate.SIGMA)

    Net.set_regularization_constant(r2)
    Net.fit(train_x, train_y, epochs, batch_size, learn_speed=learn_speed)
    Net.show_weights()
    return Net.test(test_x, test_y)


def test_iris_ENTROPY(epochs, batch_size, neurons_list, learn_speed=0.15, r2=0.1):
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

    Net = Network.Network()
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

def test_iris_with_epoch_graph():
    epochs_max = 160
    batch_size = 10
    neurons_list = [8, 3]
    learning_speed = 0.3
    r2 = 0.001
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

    Net = Network.Network()
    Net.add_loss(activate.CROSS_ENTROPY)
    for i, n in enumerate(neurons_list):
        if i == 0:
            Net.add_layer(n, activate.SIGMA, input_size=4)
        else:
            Net.add_layer(n, activate.SIGMA)

    Net.set_regularization_constant(r2)
    loss_list = []
    accuracy_list = []
    for epoch in range(epochs_max):
        Net.fit(train_x, train_y, 1, batch_size=batch_size, learn_speed=learning_speed)
        loss, accuracy = Net.test(test_x, test_y)
        loss_list.append(np.average(np.array(loss)))
        accuracy_list.append(accuracy)
    result = [(index, loss_list[index], accuracy_list[index]) for index in range(epochs_max)]
    return np.array(result)


def test_simple(epochs=10):
    x_train = []
    y_train = []
    for i in range(10):
        x_train.append((random.random(), random.random(), 1))
        y_train.append(1)
    for i in range(10):
        x_train.append((random.random(), random.random(), 0))
        y_train.append(0)

    def one_hot(a, classes : int):
        arr = np.zeros((len(a), classes))

        for i in range(len(a)):
            arr[i, a[i]] = 1
        print("One hoted", a, arr)
        return arr

    y_train = one_hot(y_train, 2)

    x_test = []
    y_test = []
    for i in range(4):
        x_test.append((random.random(), random.random(), 1))
        y_test.append(1)
    for i in range(4):
        x_test.append((random.random(), random.random(), 0))
        y_test.append(0)
    y_test = one_hot(y_test, 2)
    Net = Network.Network()
    Net.add_loss(activate.CROSS_ENTROPY)
    Net.add_layer(2, activate.SIGMA, 3)
    Net.fit(np.array(x_train), np.array(y_train), epochs)
    print(Net.test(np.array(x_test), np.array(y_test)))
    Net.show_weights()


average_loss = []
accuracy_list = []
case_loss = []
case_accuracy = []
# MSE best build
# loss, accuracy = np.array(test_iris_MSE(160, 10, [6, 3], learn_speed=0.3, r2=0.001))
# print("Total test loss:", loss, "average loss:", np.average(loss))
# average_loss.append(np.average(loss))
# accuracy_list.append(accuracy)



# loss, accuracy = np.array(test_iris_ENTROPY(160, 10, [8, 3], learn_speed=0.3, r2=0.001))
# print("Total test loss:", loss, "average loss:", np.average(loss))
# average_loss.append(np.average(loss))
# accuracy_list.append(accuracy)


# CASE 1
# average_loss = []
# accuracy_list = []
#
# loss, accuracy = np.array(test_iris_ENTROPY(160, 10, [8, 3], learn_speed=0.3, r2=0.001))
# print("Total test loss:", loss, "average loss:", np.average(loss))
# average_loss.append(np.average(loss))
# accuracy_list.append(accuracy)
#
# loss, accuracy = np.array(test_iris_MSE(240, 10, [6, 3], learn_speed=0.4, r2=0.001))
# print("Total test loss:", loss, "average loss:", np.average(loss))
# average_loss.append(np.average(loss))
# accuracy_list.append(accuracy)
#
# case_loss.append(average_loss)
# case_accuracy.append(accuracy_list)
# print("Total loss:", average_loss)
# print("Total accuracy", accuracy_list)
#
#
#
# l = np.array(case_loss)
# a = np.array(case_accuracy)
# print("Total loss cases: ", l)
# print("Total accuracy cases:", a)
# print("Best accuracy", np.average(a, axis=0))



result_array = test_iris_with_epoch_graph()
import matplotlib.pyplot as plt
def plot_loss(result_array):
    plt.plot(result_array[:, 0], result_array[:, 1], label='loss')
    # plt.plot(result_array[:, 0], result_array[:, 2], label='accuracy')
    max_epochs = result_array.shape[0]

    smoothed_loss_10 = []
    for e in range(max_epochs):
        smooth_window_10 = 10
        min_border = max(0, e - smooth_window_10)
        max_border = min(max_epochs, e + smooth_window_10)
        smoothed_loss_10.append(np.average(result_array[min_border: max_border, 1]))
    plt.plot(result_array[:, 0], np.array(smoothed_loss_10), label='smoothed by 10 loss')

    smoothed_loss_20 = []
    for e in range(max_epochs):
        smooth_window_20 = 20
        min_border = max(0, e - smooth_window_20)
        max_border = min(max_epochs, e + smooth_window_20)
        smoothed_loss_20.append(np.average(result_array[min_border: max_border, 1]))
    plt.plot(result_array[:, 0], np.array(smoothed_loss_20), label='smoothed by 20 loss')

    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.title("Loss over epochs graph")
    plt.legend()
    plt.show()


def plot_accuracy(result_array : np.ndarray):
    max_epochs = result_array.shape[0]
    # plt.plot(result_array[:, 0], result_array[:, 1], label='loss')
    plt.plot(result_array[:, 0], result_array[:, 2], label='accuracy')
    smoothed_accuracy_10 = []
    for e in range(max_epochs):
        smooth_window_10 = 10
        min_border = max(0, e - smooth_window_10)
        max_border = min(max_epochs, e + smooth_window_10)
        smoothed_accuracy_10.append(np.average( result_array[min_border : max_border, 2] ) )
    plt.plot(result_array[:, 0], np.array(smoothed_accuracy_10), label='smoothed by 10 accuracy')

    smoothed_accuracy_20 = []
    for e in range(max_epochs):
        smooth_window_20 = 20
        min_border = max(0, e - smooth_window_20)
        max_border = min(max_epochs, e + smooth_window_20)
        smoothed_accuracy_20.append(np.average(result_array[min_border: max_border, 2]))
    plt.plot(result_array[:, 0], np.array(smoothed_accuracy_20), label='smoothed by 20 accuracy')


    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.title("Accuracy over epochs graph")
    plt.legend()
    plt.show()




print("Result array: ")
print(result_array)
