import matplotlib.pyplot as plt
import mynn as nn
from draw_tools.plot import plot

import numpy as np

def visualize_conv_weights(model, conv_index=0, max_filters=16):
    """
    可视化模型中第 conv_index 个卷积层的卷积核权重
    """
    conv_count = 0
    for layer in model.layers:
        if isinstance(layer, nn.op.conv2D):
            if conv_count == conv_index:
                W = layer.params['W']  # [out_channels, in_channels, k, k]
                num_filters = min(W.shape[0], max_filters)
                fig, axes = plt.subplots(1, num_filters, figsize=(num_filters, 1.5))
                fig.suptitle(f'Conv Layer {conv_index} - First {num_filters} Filters')

                for i in range(num_filters):
                    kernel = W[i, 0]  # 只取 in_channel=0
                    axes[i].imshow(kernel, cmap='gray')
                    axes[i].axis('off')
                plt.tight_layout()
                plt.show()
                return
            conv_count += 1
    print(f"没有找到第 {conv_index} 个卷积层")

def visualize_linear_weights(model, layer_index=0, image_shape=(28, 28), max_neurons=10):
    """
    可视化全连接层的权重。每列代表一个输出神经元的权重。
    仅适用于输入是图像（可以 reshape 成 image_shape）时使用。
    """
    fc_count = 0
    for layer in model.layers:
        if isinstance(layer, nn.op.Linear):
            if fc_count == layer_index:
                W = layer.params['W']  # shape: [in_dim, out_dim]
                num_neurons = min(W.shape[1], max_neurons)
                fig, axes = plt.subplots(1, num_neurons, figsize=(num_neurons, 1.5))
                fig.suptitle(f'Linear Layer {layer_index} - First {num_neurons} Neuron Weights')
                for i in range(num_neurons):
                    weight_img = W[:, i].reshape(image_shape)
                    axes[i].imshow(weight_img, cmap='seismic', interpolation='nearest')
                    axes[i].axis('off')
                plt.tight_layout()
                plt.show()
                return
            fc_count += 1


conv_model = nn.models.Model_CNN(in_channels=1, out_channels1=3, kernel_size=3, fc_size=128, act_func='ReLU')
conv_model.load_model('best_models/best_model5.pickle')  

# 可视化第 0 个卷积层的前 8 个卷积核
visualize_conv_weights(conv_model, conv_index=0, max_filters=8)

# 可视化第 1 个卷积层的前 8 个卷积核
visualize_conv_weights(conv_model, conv_index=1, max_filters=8)

#model = nn.models.Model_MLP([784, 512, 128, 10], 'ReLU', [1e-4, 1e-4, 1e-4])
#model.load_model('best_models/best_model4.pickle')  
# 可视化第一层 Linear 的权重（输入维度为 28x28）
#visualize_linear_weights(model, layer_index=0, image_shape=(28, 28), max_neurons=10)