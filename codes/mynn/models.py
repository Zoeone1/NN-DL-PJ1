from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

class Model_CNN:
    def __init__(self, in_channels, out_channels1, kernel_size, fc_size, act_func='ReLU'):
        self.in_channels = in_channels
        self.out_channels1 = out_channels1
        self.kernel_size = kernel_size
        self.fc_size = fc_size
        self.act_func = act_func

        self.layers = []

        # Conv → ReLU → MaxPool
        self.layers.append(conv2D(in_channels=in_channels, out_channels=out_channels1, kernel_size=kernel_size))
        if act_func == 'ReLU':
            self.layers.append(ReLU())
        self.layers.append(maxpool2D(kernel_size=2, stride=2))  # 降低特征图尺寸

        # 计算 flatten 后的尺寸
        dummy_input = np.random.randn(1, in_channels, 28, 28)
        print("Dummy input shape:", dummy_input.shape)
        dummy_output = dummy_input
        for layer in self.layers:
            dummy_output = layer(dummy_output)
            print(f"Output shape after {layer.__class__.__name__}:", dummy_output.shape)
        flattened_size = np.prod(dummy_output.shape[1:])
        print("Flattened size:", flattened_size)

        # 添加 Flatten 和全连接层
        self.layers.append(Flatten())
        self.layers.append(Linear(in_dim=flattened_size, out_dim=fc_size))
        if act_func == 'ReLU':
            self.layers.append(ReLU())
        self.layers.append(Linear(in_dim=fc_size, out_dim=10))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs


    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            params = pickle.load(f)

        self.in_channels = params[0]
        self.out_channels1 = params[1]
        self.kernel_size = params[2]
        self.fc_size = params[3]
        self.act_func = params[4]

        self.layers = []
        layer_idx = 0

        # 第一个卷积层
        self.layers.append(conv2D(in_channels=self.in_channels, out_channels=self.out_channels1,
                                  kernel_size=self.kernel_size))
        self.layers[-1].W = params[layer_idx + 5]['W']
        self.layers[-1].b = params[layer_idx + 5]['b']
        self.layers[-1].weight_decay = params[layer_idx + 5]['weight_decay']
        self.layers[-1].weight_decay_lambda = params[layer_idx + 5]['lambda']
        self.layers[-1].params['W'] = self.layers[-1].W
        self.layers[-1].params['b'] = self.layers[-1].b
        expected_shape = (self.out_channels1, self.in_channels, self.kernel_size, self.kernel_size)
        assert self.layers[-1].W.shape == expected_shape, \
            f"Loaded W shape {self.layers[-1].W.shape} does not match expected {expected_shape}"
        layer_idx += 1

        if self.act_func == 'ReLU':
            self.layers.append(ReLU())

        # 池化层
        self.layers.append(maxpool2D(kernel_size=2, stride=2))

        # 重新计算卷积层输出展平后的维度
        dummy_input = np.random.randn(1, self.in_channels, 28, 28)
        print("Dummy input shape in load_model:", dummy_input.shape)
        dummy_output = dummy_input
        for layer in self.layers:
            dummy_output = layer(dummy_output)
            print(f"Output shape after {layer.__class__.__name__} in load_model:", dummy_output.shape)
        flattened_size = np.prod(dummy_output.shape[1:])
        print("Flattened size in load_model:", flattened_size)

        # 展平层
        self.layers.append(Flatten())

        # 第一个全连接层
        self.layers.append(Linear(in_dim=flattened_size, out_dim=self.fc_size))
        self.layers[-1].W = params[layer_idx + 5]['W']
        self.layers[-1].b = params[layer_idx + 5]['b']
        self.layers[-1].weight_decay = params[layer_idx + 5]['weight_decay']
        self.layers[-1].weight_decay_lambda = params[layer_idx + 5]['lambda']
        self.layers[-1].params['W'] = self.layers[-1].W
        self.layers[-1].params['b'] = self.layers[-1].b
        assert self.layers[-1].W.shape[0] == flattened_size, \
            f"Loaded W shape {self.layers[-1].W.shape[0]} does not match expected {flattened_size}"
        layer_idx += 1

        if self.act_func == 'ReLU':
            self.layers.append(ReLU())

        # 第二个全连接层
        self.layers.append(Linear(in_dim=self.fc_size, out_dim=10))
        self.layers[-1].W = params[layer_idx + 5]['W']
        self.layers[-1].b = params[layer_idx + 5]['b']
        self.layers[-1].weight_decay = params[layer_idx + 5]['weight_decay']
        self.layers[-1].weight_decay_lambda = params[layer_idx + 5]['lambda']
        self.layers[-1].params['W'] = self.layers[-1].W
        self.layers[-1].params['b'] = self.layers[-1].b
        assert self.layers[-1].W.shape[0] == self.fc_size, \
            f"Loaded W shape {self.layers[-1].W.shape[0]} does not match expected {self.fc_size}"

    def save_model(self, save_path):
        param_list = [self.in_channels, self.out_channels1, self.kernel_size, self.fc_size,
                      self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay,
                                   'lambda': layer.weight_decay_lambda})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)