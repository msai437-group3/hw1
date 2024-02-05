import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt


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


PARAMS = {
    'epoch': 100,
    'batch_size': 10,
    'learning_rate': 0.01,
    'num_hidden_layers': 100,
    'early_stopping': False
}


class MLP:

    def __init__(self, input_size, params):
        self.params = params
        self.k = self.params['num_hidden_layers']  # k nodes in the hidden layer
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

    def train(self, X_train, Y_train, X_valid, Y_valid, fig_name):
        train_losses = []  # To store training loss per epoch
        val_losses = []  # To store validation loss per epoch
        if self.params['early_stopping']:
            best_loss = float('inf')
            patience_counter = 0
            patience = 5

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
            # print(f"----------------------Epoch {i + 1}----------------------")
            # print(f"Training loss: {epoch_train_loss / total_batches}")
            train_losses.append(epoch_train_loss / total_batches)
            valid_loss = self.valid(X_valid, Y_valid)
            val_losses.append(valid_loss)
            # print(f"Validation loss: {valid_loss}")
            if self.params['early_stopping']:
                # Early stopping check
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience_counter = 0  # Reset patience since we have improvement
                    # print(f"Epoch {i}: Validation loss improved to {valid_loss:.4f}")
                else:
                    patience_counter += 1
                    # print(f"Epoch {i}: Validation loss did not improve. Patience {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print(f"Epoch {i}: Stopping early due to no improvement in validation loss.")
                        break
        # Plotting the training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.show()

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

    def plot_decision_surface(self, X_test, y_test, fig_name):
        # Set min and max values and give it some padding
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        h = 0.01  # step size in the mesh
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Forward pass to predict the class for each point in the mesh
        Z = self._forward(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the decision boundary
        plt.contour(xx, yy, Z, levels=[0.5], cmap="Greys", vmin=0, vmax=0.6)
        # Plot also the test points
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolor='k', s=20)
        plt.title('Learned Decision Surface with Test Observations')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        # plt.show()
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # train_path = 'center_surround_train.csv'
    # valid_path = 'center_surround_valid.csv'
    # test_path = 'center_surround_valid.csv'
    # train_path = 'xor_train.csv'
    # valid_path = 'xor_valid.csv'
    # test_path = 'xor_test.csv'
    # train_path = 'two_gaussians_train.csv'
    # valid_path = 'two_gaussians_valid.csv'
    # test_path = 'two_gaussians_test.csv'
    train_path = 'spiral_train.csv'
    valid_path = 'spiral_valid.csv'
    test_path = 'spiral_test.csv'
    train_data = pd.read_csv(train_path).values
    valid_data = pd.read_csv(valid_path).values
    test_data = pd.read_csv(test_path).values
    # load data
    train_data = np.random.permutation(train_data)
    valid_data = np.random.permutation(valid_data)
    X_train, Y_train = train_data[:, 1:],  train_data[:, 0]
    X_valid, Y_valid = valid_data[:, 1:],  valid_data[:, 0]
    X_test, Y_test = test_data[:, 1:],  test_data[:, 0]
    # train
    mlp = MLP(input_size=X_train.shape, params=PARAMS)
    mlp.train(X_train, Y_train, X_valid, Y_valid, train_path.split('.')[0]+'_training_validation_loss_curve.jpg')
    mlp.test(X_test, Y_test)
    mlp.plot_decision_surface(X_test, Y_test, train_path.split('.')[0]+'_decision_boundary.jpg')
    # mlp.plot()
    # experiment with different params
    # epoch = [50, 100]
    # batch_size = [10, 50, 100, 200]
    # learning_rate = [0.001, 0.01, 0.1, 0.5, 1, 2]
    # num_hidden_layers = [3, 10, 50, 100, 300]
    # early_stopping = [True, False]
    # param_combinations = list(itertools.product(epoch, batch_size, learning_rate, num_hidden_layers, early_stopping))
    # grid_search_data = []
    # for combo in param_combinations:
    #     keys = ['epoch', 'batch_size', 'learning_rate', 'num_hidden_layers', 'early_stopping']
    #     params = dict(zip(keys, combo))
    #     print (params)
    #     mlp = MLP(input_size=X_train.shape, params=params)
    #     mlp.train(X_train, Y_train, X_valid, Y_valid)
    #     accuracy = mlp.test(X_test, Y_test)
    #     params['accuracy'] = accuracy
    #     grid_search_data.append(params)
    # df = pd.DataFrame(grid_search_data)
    # df.to_csv(train_path.split('.')[0]+'_params_search.csv', index=True)






