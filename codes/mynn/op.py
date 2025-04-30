from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = np.random.randn(in_dim, out_dim) * 0.1
        self.b = np.zeros((1,out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        self.grads['W'] = self.input.T @ grad
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.params['W'].T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.randn, weight_decay=False, weight_decay_lambda=1e-4) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initialize_method = initialize_method
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        self.params = {
            'W': initialize_method(out_channels, in_channels, kernel_size, kernel_size)*0.1,
            'b': np.zeros((out_channels, 1))
        }
        
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.X_padded = None
        self.optimizable = True

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        no padding
        """
        self.input = X
        # print("forward.X", X.shape) # (4096, 1, 28, 28)
        bsz, in_channels, H, W = X.shape
        H_out = (H - self.kernel_size + 2*self.padding) // self.stride + 1
        W_out = (W - self.kernel_size + 2*self.padding) // self.stride + 1
        output = np.zeros((bsz, self.out_channels, H_out, W_out))
        # print("forward.output", output.shape) # (4096, 8, 28, 28)
        
        if self.padding > 0:
            X_padded = np.pad(X, 
                              ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              'constant')
        else:
            X_padded = X
        
        self.X_padded = X_padded
        
        for i in range(H_out):
            for j in range(W_out):
                H1 = i * self.stride
                H2 = H1 + self.kernel_size
                W1 = j * self.stride
                W2 = W1 + self.kernel_size
                
                x_slice = X_padded[:, :, H1:H2, W1:W2]
                
                for k in range(self.out_channels):
                    # print(output[:, k, i, j].shape, x_slice.shape, self.params['W'][k].shape, self.params['b'][k])
                    output[:, k, i, j] = np.sum(x_slice * self.params['W'][k, :, :, :], axis=(1, 2, 3)) + self.params['b'][k]
        
        return output
        

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        bsz, out_channels, H_out, W_out = grads.shape
        in_channels, H, W = self.input.shape[1:] # (4096, 1, 28, 28)
        # print(grads.shape, bsz, out_channels, H_out, W_out) # (4096, 8, 14, 14)
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        grad_input = np.zeros_like(self.input)
        
        X = self.input
        
        for i in range(H_out-self.padding):
            for j in range(W_out-self.padding):
                H1 = i * self.stride
                H2 = H1 + self.kernel_size
                W1 = j * self.stride
                W2 = W1 + self.kernel_size
                
                x_slice = X[:, :, H1:H2, W1:W2]
                
                for k in range(out_channels):
                    # print(i, j, k, grad_input[:, :, H1:H2, W1:W2].shape, self.params['W'][k].shape, grads[:, k, i, j][:, None, None, None].shape)
                    self.grads['b'][k] += np.sum(grads[:, k, i, j], axis=0, keepdims=True)
                    self.grads['W'][k] += np.sum(x_slice * grads[:, k, i, j][:, None, None, None], axis=0)
                    grad_input[:, :, H1:H2, W1:W2] += self.params['W'][k] * grads[:, k, i, j][:, None, None, None]
                    
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']
        
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class maxpool2D(Layer):
    """
    A simple 2D max pooling layer.
    Input shape: [batch_size, channels, H, W]
    Output shape: [batch_size, channels, H_out, W_out]
    """
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.max_indices = None  # To save the location of max for backward
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        B, C, H, W = X.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        H_out = (H - KH) // SH + 1
        W_out = (W - KW) // SW + 1
        output = np.zeros((B, C, H_out, W_out))
        self.max_indices = np.zeros_like(X, dtype=bool)

        for i in range(H_out):
            for j in range(W_out):
                h1, h2 = i * SH, i * SH + KH
                w1, w2 = j * SW, j * SW + KW
                region = X[:, :, h1:h2, w1:w2]
                max_val = np.max(region, axis=(2, 3), keepdims=True)
                output[:, :, i, j] = max_val.squeeze(-1).squeeze(-1)
                self.max_indices[:, :, h1:h2, w1:w2] += (region == max_val)

        return output

    def backward(self, grad):
        B, C, H, W = self.input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        H_out = (H - KH) // SH + 1
        W_out = (W - KW) // SW + 1
        grad_input = np.zeros_like(self.input)

        for i in range(H_out):
            for j in range(W_out):
                h1, h2 = i * SH, i * SH + KH
                w1, w2 = j * SW, j * SW + KW
                mask = self.max_indices[:, :, h1:h2, w1:w2]
                grad_block = grad[:, :, i, j][:, :, None, None]
                grad_input[:, :, h1:h2, w1:w2] += grad_block * mask

        return grad_input

        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Flatten(Layer):
    """
    A flattening layer to reshape 4D tensor [B, C, H, W] into 2D tensor [B, C*H*W]
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape  # Save shape for backward
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        if self.has_softmax:
            predicts = softmax(predicts)
        batch_size = predicts.shape[0]
        log_probs = -np.log(np.clip(predicts[np.arange(batch_size), labels], 1e-10, 1.0))  # 避免log(0)
        loss = np.mean(log_probs)
        self.grads = predicts.copy()
        self.grads[np.arange(batch_size), labels] -= 1
        self.grads /= batch_size
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model, weight_decay_lambda = 1e-8):
        super().__init__()
        self.model = model
        self.weight_decay_lambda = weight_decay_lambda

    def forward(self):
        l2_loss = 0
        for layer in self.model.layers:
            if hasattr(layer, 'W'):
                l2_loss += np.sum(np.square(layer.W))
        l2_loss *= 0.5 * self.weight_decay_lambda

        return l2_loss
    
    def backward(self): # 直接更新梯度，不用返回值
        for layer in self.model.layers:
            if hasattr(layer, 'W'):
                layer.grads['W'] += self.weight_decay_lambda * layer.W
            

class CombinedLoss:
    def __init__(self, model, max_classes, weight_decay_lambda):
        self.cross_entropy_loss = MultiCrossEntropyLoss(model=model, max_classes=max_classes)
        self.l2_regularization = L2Regularization(model=model, weight_decay_lambda=weight_decay_lambda)

    def __call__(self, predicts, labels):
        ce_loss = self.cross_entropy_loss(predicts, labels)
        l2_loss = self.l2_regularization.forward()
        return ce_loss + l2_loss

    def backward(self):
        self.cross_entropy_loss.backward()
        self.l2_regularization.backward()


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

#You're welcome to implement other complicated layer (e.g.  ResNet Block or Bottleneck)