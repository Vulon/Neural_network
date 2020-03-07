import random
import numpy as np
import activate


class Layer:
    def __init__(self, neurons : int, input_size, activation_function : int):
        self.weights = np.random.rand(neurons, input_size)
        self.bias = np.random.rand(neurons, 1)
        self.activate_num = activation_function

        self.activate_func = activate.functions[activation_function]
        self.activate_der = activate.derivatives[activation_function]

        self.input_size = input_size
        self.neurons = neurons

        self.sum = None
        self.activated_sum = None
        self.input = None


    def forward(self, input : np.ndarray, verbose=0):
        if verbose:
            print("Layer forward. Input=", input)
        self.input = input
        product = np.dot(self.weights, input)
        if verbose:
            print("Weights: ", self.weights)
            print("Product: ", product)
        product.resize((product.size, 1))
        product += self.bias
        if verbose:
            print("Bias: ", self.bias)
            print("Product: ", product)
        self.sum = product.copy()
        product = self.activate_func(product).copy()
        if verbose:
            print("Activated Product: ", product)
            print("Layer forward complete __________")
        self.activated_sum = product.copy()
        return product
    def get_derivative_for_last_sum(self, verbose=False):
        if verbose:
            print("Layer sum: ", self.sum)
            print("Activations f. derivative: ", self.activate_der(self.sum))
        res = self.activate_der(self.sum)
        return res

    def get_weights_T(self):
        return self.weights.T

    def optimize(self, error : np.ndarray, learn_speed=0.1, verbose=False, regularization_constant=0):
        if verbose:
            print("Optimize layer")
            print("weights: ", self.weights)
            print("bias: ", self.bias)
            print("error: ", error)
            print("Input", self.input)
            print("Self activated sum", self.activated_sum)
            print("Changing bias by ", - error * learn_speed - self.bias * regularization_constant)
            print("Changing weights by ", -learn_speed * np.dot(error, self.input.T) - self.weights * regularization_constant)

        self.bias = self.bias - error * learn_speed - self.bias * regularization_constant
        self.weights = self.weights - learn_speed * np.dot(error, self.input.T) - self.weights * regularization_constant
        if verbose:
            print("New bias", self.bias)
            print("New weights: ", self.weights)
                


class Network:    

    def add_layer(self, neurons_count : int, activate_function : int, input_size=None):
        if input_size is None:
            input_size = self.layers[len(self.layers) - 1].neurons
        self.layers.append(Layer(neurons_count, input_size, activate_function))
        

    def __init__(self, verbose=0):
        self.layers = []
        self.loss = None
        self.loss_der = None
        self.verbose = verbose
        self.regularization_constant = 0

    def add_loss(self, loss_function : int):
        self.loss = activate.loss[loss_function]
        self.loss_der = activate.loss_der[loss_function]

    def set_regularization_constant(self, value : float):
        self.regularization_constant = value
    
    def size(self, ):
        return len(self.layers)

    def batch_forward(self, input : np.ndarray):
        result = []
        for row in input:
            result.append(self.forward(row.T).T)
        return np.array(result)

    def forward(self, input : np.ndarray):
        if self.verbose:
            print("Net forward input", input, "Input shape", input.shape)

        input.resize((input.size, 1))
        for l in self.layers:
            input = l.forward(input, self.verbose)
        return input

    def test(self, test_x : np.ndarray, test_y : np.ndarray):
        loss = []
        true = 0
        count = 0
        if test_x.shape[0] > 1:
            predicted = self.batch_forward(test_x)

            for i, entry in enumerate( predicted):
                print("Predicted entry", i, entry)
                print("True y", test_y[i])
                print("Loss", self.loss(test_y[i], entry))
                loss.append(self.loss(test_y[i], entry))
                if np.argmax(entry) == np.argmax(test_y[i]):
                    true += 1
                count += 1

        else:
            predicted = self.forward(test_x)
            loss = self.loss(test_y, predicted)
            print("Predicted", predicted)
            print("True", test_y)
            print("loss", loss)
            if np.argmax(test_y) == np.argmax(predicted):
                true += 1
            count += 1
        return loss, true / count


    def show_weights(self):
        weights = []
        bias = []
        for l in self.layers:
            weights.append(l.weights)
            bias.append(l.bias)
        print("Weights", weights)
        print("Bias", bias)
        return weights
    
    def fit(self, x :np.ndarray, y : np.ndarray, epochs :int, batch_size=1, learn_speed=0.1):
        
        batch_count = int(y.shape[0]  / batch_size + (y.shape[0]  % batch_size > 0))
        epoch_loss =[]
        for ep in range(epochs):
            loss = []
            
            for i in range(batch_count):
                batch_y = y[i * batch_size : min(y.shape[0], batch_size * i + batch_size)  ]
                batch_x = x[i * batch_size : min(x.shape[0], batch_size * i + batch_size)  ]
    
                permutations = [t for t in range(batch_y.shape[0])]
                random.shuffle(permutations)
                permutations = np.array(permutations)
                batch_x = batch_x[permutations]
                batch_y = batch_y[permutations]
    
                for input_index in range(batch_x.shape[0]):
                    error = [None for t in self.layers]
                    inp = batch_x[input_index]
                    y_true = batch_y[input_index]
                    y_true.resize((y_true.size, 1))
                    inp.resize((inp.size, 1))
                    inp = self.forward(inp)
                    loss.append(self.loss(y_true, inp))
                    if self.verbose:
                        print("Count last layer error")
                        print("Loss between derivative", y_true, inp, "is", self.loss_der(y_true, inp))
                        print("Last layer function derivative for last sum is", self.layers[-1].get_derivative_for_last_sum())
                        print("Last layer error is: ", self.loss_der(y_true, inp) * self.layers[-1].get_derivative_for_last_sum(self.verbose))

                    error[ -1] = self.loss_der(y_true, inp) * self.layers[-1].get_derivative_for_last_sum(self.verbose)

                    for e_index in range(len(error) - 2, -1, -1):
                        dot_product = np.dot(self.layers[e_index + 1].get_weights_T(), error[e_index + 1])
                        layer_derivative = self.layers[e_index].get_derivative_for_last_sum(self.verbose)
                        error[e_index] = dot_product * layer_derivative
    
                    for e_index in range(len(error)):
                        self.layers[e_index].optimize(error[e_index], learn_speed, verbose=self.verbose, regularization_constant=self.regularization_constant)
            
            epoch_loss.append(np.average(np.array(loss)))
            print("Epoch", ep, "Loss", epoch_loss[-1])


