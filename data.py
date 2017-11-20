
import numpy as np

def get_dataset():
    """
    获取线性数据集
    :return:返回x,y的数据
    """
    x = np.random.rand(100).astype(np.float32)
    y = 4.8 *x + 8.5
    return x,y