
import numpy as np

def get_dataset():
    """
    获取线性训练数据集和测试数据集
    :return:x_train_data，y_train_data 训练数据集
            x_test_data，y_test_data 测试数据集
    """
    # 设置训练数据集
    x_train_data = np.random.rand(1000).astype(np.float32)
    y_train_data = 4.8 *x_train_data + 8.5

    #设置测试数据集
    x_test_data = np.random.rand(50).astype(np.float32)
    y_test_data = 4.8 *x_test_data + 8.5

    return x_train_data,y_train_data,x_test_data,y_test_data