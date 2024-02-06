import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Iterable
import wandb


def make_data(name: str, path: str, scale: bool=True):
    data_all = []
    for dtsplit in ['train', 'valid', 'test']:
        dtpath = os.path.join(path, f"{name}_{dtsplit}.csv")
        data = pd.read_csv(dtpath).to_numpy()
        if scale:
            _max, _min = np.max(data[:, 1:], axis=0, keepdims=True), np.min(data[:, 1:], axis=0, keepdims=True)
            data[:, 1:] = (data[:, 1:] - _min) / (_max - _min)
        data_all.append(data)
    return data_all

def accuracy_score(preds: np.ndarray, labels: np.ndarray):
    return np.mean(((preds>0.5).astype('int32')==labels.astype('int32')).astype('float32'))


class MLP(object):
    def __init__(self, ly1=[2, 5], ly2=[5, 1], lr=1e-2, momentum=0.9, criterian='bce') -> None:
        self.momentum = momentum
        self.lr = lr
        self.criterian = criterian
        self.w1 = np.random.randn(ly1[0], ly1[1]) * 0.01
        self.w2 = np.random.randn(ly2[0], ly2[1]) * 0.01
        self.b1 = np.zeros((1, ly1[1]))
        self.b2 = np.zeros((1, ly2[1]))
        self.w1grad = np.zeros_like(self.w1)
        self.w2grad = np.zeros_like(self.w2)
        self.b1grad = np.zeros_like(self.b1)
        self.b2grad = np.zeros_like(self.b2)
        
    def predict(self, x: np.ndarray):
        return self._forward(x)
    
    def train_single_iteration(self, x: np.ndarray, y: np.ndarray):
        p, loss, (w1grad, w2grad, b1grad, b2grad) = self._forward_and_backward(x, y)
        self._optimize_sgd_with_momentum(w1grad, w2grad, b1grad, b2grad)
        return p, loss
        
    def _forward(self, x: np.ndarray):
        z1 = self._linear_matmul(x, self.w1)
        z1b = self._linear_addbias(z1, self.b1)
        o1 = self._relu(z1b)
        z2 = self._linear_matmul(o1, self.w2)
        z2b = self._linear_addbias(z2, self.b2)
        p = self._sigmoid(z2b)
        return p
    
    def _forward_and_backward(self, x: np.ndarray, y: np.ndarray):
        # forward pass
        z1 = self._linear_matmul(x, self.w1)
        z1b = self._linear_addbias(z1, self.b1)
        o1 = self._relu(z1b)
        z2 = self._linear_matmul(o1, self.w2)
        z2b = self._linear_addbias(z2, self.b2)
        p = self._sigmoid(z2b)
        loss = self._bce(p, y) if self.criterian=='bce' else self._mse(p, y)
        
        # backward pass
        grad = self._mean_backward(np.ones((1,)), p.shape[0])
        if self.criterian == 'bce':
            grad = self._bce_with_sigmoid_backward(grad, p, y)
        else:
            grad = self._mse_backward(grad, p, y)
            grad = self._sigmoid_backward(grad, p)
        b2grad= self._linear_b_backward(grad)
        w2grad = self._linear_w_backward(grad, o1)
        grad = self._linear_x_backward(grad, self.w2)
        grad = self._relu_backward(grad, z1)
        b1grad = self._linear_b_backward(grad)
        w1grad = self._linear_w_backward(grad, x)
        
        return p, loss, (w1grad, w2grad, b1grad, b2grad)
    
    def _optimize_sgd_with_momentum(self, w1grad, w2grad, b1grad, b2grad):
        self.w1grad = self.momentum * self.w1grad + (1-self.momentum) * w1grad
        self.w2grad = self.momentum * self.w2grad + (1-self.momentum) * w2grad
        self.b1grad = self.momentum * self.b1grad + (1-self.momentum) * b1grad
        self.b2grad = self.momentum * self.b2grad + (1-self.momentum) * b2grad
        self.w2 -= self.lr * self.w2grad
        self.w1 -= self.lr * self.w1grad
        self.b2 -= self.lr * self.b2grad
        self.b1 -= self.lr * self.b1grad
    
    # Linear forward / backward
    def _linear_matmul(self, x: np.ndarray, w: np.ndarray,):
        # x: [N, C1]; w: [C1, C2]
        return np.matmul(x, w)
    
    def _linear_addbias(self, x: np.ndarray, b: np.ndarray):
        # x: [N, C1]; b: [1, C2]
        return x + b
    
    def _linear_x_backward(self, grad: np.ndarray, w: np.ndarray):
        # grad: [N, C2]; w: [C1, C2]
        return np.matmul(grad, w.T)
    
    def _linear_w_backward(self, grad: np.ndarray, x: np.ndarray):
        # grad: [N, C2]; x: [N, C1]
        return np.matmul(x.T, grad)
    
    def _linear_b_backward(self, grad: np.ndarray):
        # grad: [N, C2]
        return np.sum(grad, axis=0, keepdims=True)
    
    # ReLU forward / backward
    def _relu(self, x: np.ndarray):
        return np.where(x<0, 0, x)
    
    def _relu_backward(self, grad: np.ndarray, x: np.ndarray):
        return np.where(x<0, 0, 1) * grad
    
    # Sigmoid forward / backward
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_backward(self, grad, o):
        return grad * o * (1 - o)
    
    # BCE forward / backward
    def _bce(self, p: np.ndarray, y: np.ndarray, epsilon: float=1e-6):
        p = np.clip(p, epsilon, 1 - epsilon)  # avoid log(0)
        return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))
    
    def _bce_with_sigmoid_backward(self, grad: np.ndarray, p: np.ndarray, y: np.ndarray):
        # p: [N,1]; y: [N,1]
        return grad * (p - y)
    
    # MSE forward / backward
    def _mse(self, p: np.ndarray, y: np.ndarray):
        return np.mean(np.power(p-y, 2))
    
    def _mse_backward(self, grad: np.ndarray, p: np.ndarray, y: np.ndarray):
        return grad * 2 * (p - y)
    
    # Mean backward
    def _mean_backward(self, grad: np.ndarray, N: int):
        return (1/N) * grad


def _train_mlp(
    data: Iterable, hidden_nums: int=10, lr: float=1e-1, bs: int=20, 
    epochs: int=100, criterian: str='bce', momentum: float=0.9,
    eval_interval: int=5, shuffle: bool=False):
    
    model = MLP(ly1=[2, hidden_nums], ly2=[hidden_nums, 1], lr=lr, momentum=momentum, criterian=criterian)
    train_data, valid_data = data
    
    for e in range(epochs):
        if shuffle:
            np.random.shuffle(train_data)
        for i in range(train_data.shape[0] // bs):
            batch_inputs = train_data[i*bs:(i+1)*bs, 1:]
            batch_labels = train_data[i*bs:(i+1)*bs, 0:1]
            
            p, loss = model.train_single_iteration(batch_inputs, batch_labels)

        wandb.log({                         
            'epoch': e+1, 
            'val_acc': accuracy_score(model.predict(valid_data[:, 1:]), valid_data[:, 0:1])})
    

def train_wandb_xor():
    train_data, valid_data, test_data = make_data('xor', "./datasets", False)
    with wandb.init():
        config = wandb.config
        _train_mlp(
            (train_data, valid_data),
            hidden_nums=config.hidden_size,
            lr=config.learning_rate,
            bs=config.batch_size,
            epochs=config.epochs,
            criterian=config.objective_func)

def train_wandb_center_surround():
    train_data, valid_data, test_data = make_data('center_surround', "./datasets", False)
    with wandb.init():
        config = wandb.config
        _train_mlp(
            (train_data, valid_data),
            hidden_nums=config.hidden_size,
            lr=config.learning_rate,
            bs=config.batch_size,
            epochs=config.epochs,
            criterian=config.objective_func)

def train_wandb_spiral():
    train_data, valid_data, test_data = make_data('spiral', "./datasets", False)
    with wandb.init():
        config = wandb.config
        _train_mlp(
            (train_data, valid_data),
            hidden_nums=config.hidden_size,
            lr=config.learning_rate,
            bs=config.batch_size,
            epochs=config.epochs,
            criterian=config.objective_func)

def train_wandb_two_gaussians():
    train_data, valid_data, test_data = make_data('two_gaussians', "./datasets", False)
    with wandb.init():
        config = wandb.config
        _train_mlp(
            (train_data, valid_data),
            hidden_nums=config.hidden_size,
            lr=config.learning_rate,
            bs=config.batch_size,
            epochs=config.epochs,
            criterian=config.objective_func)


if __name__ == "__main__":
    wandb.login(key="<Your API Key>")
    np.random.seed(0)

    sweep_config = {
        'method': 'grid'
    }
    metric = {
        'name': 'val_acc',
        'goal': 'maximize'   
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        'hidden_size': {
            'values': [2, 16, 64]
        },
        'learning_rate': {
            'values': [1, 1e-1, 1e-2, 1e-3]
        },
        'batch_size': {
            'values': [1, 5, 25, 200]
        },
        'epochs': {
            'values': [50, 100]
        },
        'objective_func': {
            'values': ['bce', 'mse']
        }
    }
    sweep_config['parameters'] = parameters_dict


    sweep_id = wandb.sweep(sweep_config, project="MSAI437-HW1")

    # wandb.agent(sweep_id, train_wandb_xor)
    # wandb.agent(sweep_id, train_wandb_center_surround)
    # wandb.agent(sweep_id, train_wandb_spiral)
    wandb.agent(sweep_id, train_wandb_two_gaussians)