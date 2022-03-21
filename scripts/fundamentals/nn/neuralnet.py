# imports
import numpy as np

# Class
class NeuralNet:
    # Constructor
    def __init__(self, layers, alpha=0.1):
        # Init weights and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # Fill in weights list
        for i in np.arange(0, len(layers) - 2):
            # Randomly initialize weights
            # and add extra node for bias
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # Special case for last layer
        # input last layer needs bias but output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    # Useful for debuggind
    def __repr__(self):
        # Construct a string that represents nn architecture and return
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    # Define Sigmoid Activation
    def sigmoid(self, x):
        # Return sigmoid activation
        return 1.0 / (1+np.exp(-x))

    # Define derivative function
    def sigmoid_deriv(self, x):
        # Compute derivative and return
        return x * (1-x)

    # Create function named 'fit' to train the model
    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # Insert bias columns of ones
        X = np.c_[X, np.ones((X.shape[0]))]

        # Loop over all epochs
        for epoch in np.arange(0, epochs):
            # Loop over individual data point
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # Check to display training update to terminal
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1, loss))

    # Fit partial function
    def fit_partial(self, x, y):
        # Construct the list of output activations for each layer
        # First activation is a special case, it's just input feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDWORWARD
        # Now we can start forward propagation phase
        for layer in np.arange(0, len(self.W)):
            # Feedforward the activation of current layer by
            # taking dot product (called "net input" to current layer)
            net = A[layer].dot(self.W[layer])

            # Compute "net output" by applying non-linear activation function
            out = self.sigmoid(net)

            # Append to list of activations
            A.append(out)

        # BACKPROPAGATION
        # The first phase is to compute the difference between our *prediction* and
        # and the true target value
        error = A[-1] - y

        # Now we need to apply the chain rule and build
        # our list of deltas 'D'. The first entry in 'D' is the error of the
        # output layer times the derivative of the activation function for the output value
        D = [error * self.sigmoid(A[-1])]

        # Given the delta for the final layer in the network, we can work backward using a for loop
        # Simply loop over the layers in reverse order (ignoring the last two since we have
        # already taken them into account)
        for layer in np.arange(len(A)-2, 0, -1):
            # The delta for the current layer is equal to the delta of the previous layer dotted
            # with the weight matrix of the current layer, followed by multiplying the delta
            # by the derivative of the nonlinear activation function for the activations of the current layer.
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            # Append
            D.append(delta)

        # Given our delta list D, we can move on to the weight update phase
        # Since we looped on layers in reverse order, lets flip the order of D
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        # Loop over all layers
        for layer in np.arange(0, len(self.W)):
            # Update the weights by taking the dot product of the layer activations
            # with respective deltas, then multiplying by alpha and adding to weight matrix.
            # THIS IS WHERE THE ACTUAL LEARING TAKES PLACE
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    # Prediction function (given trained model)
    def predict(self, X, addBias = True):
        # Initialize the output prediction as input features
        p = np.atleast_2d(X)

        # Check if to add bias
        if addBias:
            # Add 1 column for bias
            p = np.c_[p, np.ones((p.shape[0]))]

        # Loop over layers in network
        for layer in np.arange(0, len(self.W)):
            # Compute output prediction is just taking the dot product of the current activation value 'p'
            # and the weight matrix of the current layer, then passing it through nonlinear activation function
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # Return predicted value
        return p

    # Calculate loss function
    def calculate_loss(self, X, targets):
        # Make predictions for the input data and compute loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        # return loss
        return loss

