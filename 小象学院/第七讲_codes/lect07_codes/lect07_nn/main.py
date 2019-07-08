# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/13
    项目名称： 人工神经网络 模型的手工实现
    参考：    https://github.com/mattm/simple-neural-network
"""
from neural_network import NeuralNetwork


def run_main():
    """
        主函数
    """

    # 输入层：2个节点
    # 隐含层：2个节点
    # 输出层：2个节点
    # 初始化各层的权重和偏置
    nn = NeuralNetwork(2, 2, 2,
                       hidden_layer_weights=[0.15, 0.2, 0.25, 0.3],
                       hidden_layer_bias=0.35,
                       output_layer_weights=[0.4, 0.45, 0.5, 0.55],
                       output_layer_bias=0.6)

    # 训练网络（这里只用一个样本作为训练作为示例）
    sample_input = [0.05, 0.1]
    sample_output = [0.01, 0.99]
    # 迭代10000次训练网络
    print('训练神经网络')
    for i in range(10000):
        # 提供输入和输出
        nn.fit(sample_input, sample_output)
        # 输出误差
        # 会看到随着迭代的增多，误差是变小的
        print('{}次迭代，误差：{}'.format(i + 1, nn.calculate_total_error([[sample_input, sample_output]])))

    # 测试网络（这里选取一个和训练样本接近的样本）
    print('测试神经网络:')
    test_sample = [0.049, 0.11]

    print('训练样本:', sample_input, sample_output)
    print()
    print('输入样本:', test_sample)
    print('预测结果:', nn.feed_forward(test_sample))


if __name__ == '__main__':
    run_main()