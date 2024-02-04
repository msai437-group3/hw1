import pandas as pd
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-7  # Small constant to prevent division by zero
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    return loss


def add_bias(X):
    # Add a column of ones to the input X
    ones = np.ones((X.shape[0], 1))
    return np.hstack([X, ones])


class MLP:

    def __init__(self, input_size, k=3):
        self.params = {
            'epoch': 100,
            'batch_size': 10,
            'learning_rate': 0.1
        }
        self.k = k  # k nodes in the hidden layer
        # self.w = np.zeros((input_size[1] + 1, self.k))  # +1 bias weight
        # self.v = np.zeros((self.k + 1, output_size))  # +1 bias weight
        self.input_size = input_size
        self.w = np.random.rand(input_size[1] + 1, self.k)
        self.v = np.random.rand(self.k + 1, 1)

    def _forward(self, X):
        # layer 1 - add bias term
        self.X = add_bias(X)
        # layer 1 - linear layer
        self.z1 = np.dot(self.X, self.w)
        # layer 1 - activation layer
        self.a1 = sigmoid(self.z1)
        # layer 2 - add bias term
        self.a1_with_bias = add_bias(self.a1)
        # layer 2 - linear layer
        self.z2 = np.dot(self.a1_with_bias, self.v)
        # layer 2 - activation layer
        self.a2 = sigmoid(self.z2)
        return self.a2

    def _backward(self, Y):
        grad_v = np.dot((self.a2 - Y.reshape(-1, 1)).T, self.a1_with_bias)
        grad_w = np.dot(self.a2 - Y.reshape(-1, 1), self.v.T)
        grad_w = grad_w * derivative_sigmoid(self.a1_with_bias)
        grad_w = np.dot(grad_w[:, :-1].T, self.X)  # Exclude gradient for the bias for the next layer

        self.v -= self.params['learning_rate'] * grad_v.T
        self.w -= self.params['learning_rate'] * grad_w.T

    def train(self, X_train, Y_train, X_valid, Y_valid):
        for i in range(self.params['epoch']):
            total_batches = int(np.ceil(X_train.shape[0] / self.params['batch_size']))
            epoch_train_loss = 0
            for j in range(total_batches):
                start_idx = j * self.params['batch_size']
                end_idx = min(start_idx + self.params['batch_size'], X_train.shape[0])
                X_train_batch = X_train[start_idx:end_idx]
                Y_train_batch = Y_train[start_idx:end_idx]
                output = self._forward(X_train_batch)
                batch_train_loss = binary_cross_entropy(Y_train_batch, np.squeeze(output))
                epoch_train_loss += batch_train_loss
                self._backward(Y_train_batch)
            print(f"----------------------Epoch {i + 1}----------------------")
            print(f"Training loss: {epoch_train_loss / total_batches}")
            valid_loss = self.valid(X_valid, Y_valid)
            print(f"Validation loss: {valid_loss}")

    def valid(self, X, Y):
        output = self._forward(X)
        loss = binary_cross_entropy(Y, np.squeeze(output))
        return loss

    def predict(self, X):
        output = self._forward(X)
        prediction = np.where(output >= 0.5, 1.0, 0.0)
        return prediction.squeeze()

    def test(self, X, Y):
        prediction = self.predict(X)
        accuracy = np.mean(prediction == Y)
        print(f"Test accuracy: {accuracy}")
        return accuracy

    def plot(self):
        # plot training loss, validation loss, test loss
        pass


if __name__ == '__main__':
    train_path = 'center_surround_train.csv'
    valid_path = 'center_surround_valid.csv'
    test_path = 'center_surround_valid.csv'
    # train_path = 'xor_train.csv'
    # valid_path = 'xor_valid.csv'
    # test_path = 'xor_test.csv'
    # train_path = 'two_gaussians_train.csv'
    # valid_path = 'two_gaussians_valid.csv'
    # test_path = 'two_gaussians_test.csv'
    # train_path = 'spiral_train.csv'
    # valid_path = 'spiral_valid.csv'
    # test_path = 'spiral_test.csv'
    train_data = pd.read_csv(train_path).values
    valid_data = pd.read_csv(valid_path).values
    test_data = pd.read_csv(test_path).values
    # load data
    train_data = np.random.permutation(train_data)
    valid_data = np.random.permutation(valid_data)
    X_train, Y_train = train_data[:, 1:],  train_data[:, 0]
    X_valid, Y_valid = valid_data[:, 1:],  valid_data[:, 0]
    X_test, Y_test = test_data[:, 1:],  test_data[:, 0]
    # start training
    mlp = MLP(input_size=X_train.shape)
    mlp.train(X_train, Y_train, X_valid, Y_valid)
    mlp.test(X_test, Y_test)
    # mlp.plot()
