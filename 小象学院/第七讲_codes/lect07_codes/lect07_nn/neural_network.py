# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/13
    项目名称： 人工神经网络 模型的手工实现
    参考：    https://github.com/mattm/simple-neural-network
"""

import random
import math

#
# "pd_"     : 偏导数
# "d_"      : 导数
# "_wrt_"   : "with respect to" 对于变量
# "w_ho"    : 由隐含层(h)到输出层(o)的weights的索引
# "w_ih"    : 由输入层(i)到隐含层(h)的weights的索引


class NeuralNetwork(object):
    """
        神经网络类
    """
    LEARNING_RATE = 0.5  # 学习率

    def __init__(self,
                 num_inputs,                    # 输入层神经元个数
                 num_hidden,                    # 隐含层神经元个数
                 num_outputs,                   # 输出层神经元个数
                 hidden_layer_weights=None,     # 隐含层神经元权重
                 hidden_layer_bias=None,        # 隐含层神经元偏置
                 output_layer_weights=None,     # 输出层神经元权重
                 output_layer_bias=None         # 输出层神经元偏置
                 ):
        # 输入层
        self.num_inputs = num_inputs

        # 隐含层
        self.hidden_layer = Layer(num_hidden, hidden_layer_bias)

        # 输出层
        self.output_layer = Layer(num_outputs, output_layer_bias)

        # 初始化各层权重
        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        """
            初始化从输入层到隐含层的权重
        """
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        """
            初始化从隐含层到输出层的权重
        """
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        """
            信息输出
        """
        print('------')
        print('* 输入层: {}'.format(self.num_inputs))
        print('------')
        print('隐含层')
        self.hidden_layer.inspect()
        print('------')
        print('* 输出层')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        """
            前向传播
        """
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def fit(self, training_inputs, training_outputs):
        """
            根据输入输出训练网络
        """
        # 前向传播
        self.feed_forward(training_inputs)

        # 1. 输出层神经元error计算
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # 求偏导
            pd_errors_wrt_output_neuron_total_net_input[o] = \
                self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. 隐含层神经元erro计算
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += \
                    pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            pd_errors_wrt_hidden_neuron_total_net_input[h] = \
                d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. 更新输出层权重
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight = \
                    pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. 更新隐含层权重
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        """
            计算整体误差
        """
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error


class Layer:
    """
        网络层类
    """
    def __init__(self, num_neurons, bias):

        # 同一层神经元的偏置都一致
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        """
            信息输出
        """
        print('神经元:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' 神经元', n)
            for w in range(len(self.neurons[n].weights)):
                print('  权重:', self.neurons[n].weights[w])
            print('  偏置:', self.bias)

    def feed_forward(self, inputs):
        """
            前向传播
        """
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        """
            获取输出结果
        """
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron(object):
    """
        神经元节点
    """
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        """
            计算输出
        """
        self.inputs = inputs
        # 激活函数
        self.output = self.activate(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        """
            计算整体输入
        """
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def activate(self, total_net_input):
        """
            逻辑回归激活函数
        """
        return 1 / (1 + math.exp(-total_net_input))

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        """
            计算相对于整体网络输入的误差偏导数
        """
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        """
            计算每个神经元节点的误差（均方误差)
        """
        return 0.5 * (target_output - self.output) ** 2

    def calculate_pd_error_wrt_output(self, target_output):
        """
            计算相对于输出的误差偏导数
        """
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_input(self):
        """
            计算相对于输入的总体偏导数
        """
        return self.output * (1 - self.output)

    def calculate_pd_total_net_input_wrt_weight(self, index):
        """
            计算相对于权重的整体输入的偏导数
        """
        return self.inputs[index]