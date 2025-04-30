from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer): #基于动量的梯度下降优化算法
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu  # 动量系数
        self.velocity = {}  # 用于存储每个可优化层参数的速度
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[layer] = {}
                for key in layer.params.keys():
                    self.velocity[layer][key] = np.zeros_like(layer.params[key])

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    # 更新速度
                    self.velocity[layer][key] = self.mu * self.velocity[layer][key] - self.init_lr * layer.grads[key]
                    # 更新参数
                    layer.params[key] += self.velocity[layer][key]