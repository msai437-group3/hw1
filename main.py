# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, nodes, output_size, dropout_prob):
        # random initialization of the weights
        self.weights_hidden = np.random.randn(input_size, nodes)
        self.bias_hidden = np.zeros((1, nodes))
        # for i in range(1, len(hidden_size)):
        #     self.weights_hidden.append(np.random.randn(hidden_size[i - 1], hidden_size[i]))
        #     self.bias_hidden.append(np.zeros((1, hidden_size[i])))

        # Initialize weights and biases for the output layer
        self.weights_output = np.random.randn(nodes, output_size)
        self.bias_output = np.zeros((1, output_size))

        # Dropout probability
        self.dropout_prob = dropout_prob

        # Mask for dropout (initialized as None)
        self.dropout_mask = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def dropout(self, x):
        # Generate a binary mask with the specified dropout probability
        self.dropout_mask = (np.random.rand(*x.shape) < self.dropout_prob) / (1 - self.dropout_prob)
        return x * self.dropout_mask

    def binary_cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15  # Small constant to avoid numerical instability in log
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)

    def forward(self, X, training=False):
        # Forward pass through the hidden layer
        hidden_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        # Apply dropout during training
        if training:
            hidden_output = self.dropout(hidden_output)
        # Forward pass through the output layer
        output_input = np.dot(hidden_output, self.weights_output) + self.bias_output
        output = self.sigmoid(output_input)

        return hidden_output, output

    def train(self, X, y, learning_rate, epochs, lossFunction):
        for epoch in range(epochs):
            # Forward pass
            hidden_output, output = self.forward(X)
            if lossFunction == "MSE":
                # Calculate the loss (mean squared error)
                loss = 0.5 * np.mean((output - y) ** 2)
            elif lossFunction == "BCE":
                # Calculate the loss (Binary cross entropy)
                loss = self.binary_cross_entropy_loss(y, output)

            # Backpropagation
            output_error = output - y
            output_delta = output_error * self.sigmoid_derivative(output)

            hidden_error = output_delta.dot(self.weights_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

            # Update weights and biases
            self.weights_output -= learning_rate * hidden_output.T.dot(output_delta)
            self.bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

            self.weights_hidden -= learning_rate * X.T.dot(hidden_delta)
            self.bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')


def print_model_accuracy(train_data, test_data):
    # Extracting train x and y labels
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    # Extracting test x and y labels
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    # Assuming you have input data X and target labels y
    input_size = X_train.shape[1]
    output_size = 1  # For binary classification
    nodes = 50  # Adjust as needed
    learning_rate = 0.1
    epochs = 10000
    lossFunction = "BCE"
    # Create and train the neural network with dropout
    dropout_prob = 0.9  # Adjust the dropout probability as needed
    model = NeuralNetwork(input_size, nodes, output_size, dropout_prob)
    model.train(X_train, y_train.reshape(-1, 1), learning_rate, epochs, lossFunction)
    # Make predictions
    hidden_output, predictions = model.forward(X_test)
    binary_predictions = (predictions > 0.5).astype(int)
    # Calculate and print various classification metrics
    accuracy = accuracy_score(y_test.astype(int), binary_predictions)
    # Use a breakpoint in the code line below to debug your script.
    print(f'Accuracy: {accuracy:.4f}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Example usage:
    # Assuming your data is saved in a CSV file named 'your_dataset.csv'
    # filename1 = 'center_surround_train.csv'
    # filename2 = 'center_surround_test.csv'
    # Accuracy: 0.7400
    # filename1 = 'spiral_train.csv'
    # filename2 = 'spiral_test.csv'
    # Accuracy: 0.8000
    # filename1 = 'two_gaussians_train.csv'
    # filename2 = 'two_gaussians_test.csv'
    # Accuracy: 0.9050
    filename1 = 'xor_train.csv'
    filename2 = 'xor_test.csv'
    # Accuracy: 0.9950
    # Load the data using np.loadtxt, skipping the first row
    train_data = np.loadtxt(filename1, delimiter=',', skiprows=1)
    test_data = np.loadtxt(filename2, delimiter=',', skiprows=1)

    print_model_accuracy(train_data,test_data)


