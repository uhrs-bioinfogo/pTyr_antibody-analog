import numpy as np
from tensorflow.keras import layers, Sequential, losses, optimizers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# 配置GPU内存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    model = tf.keras.models.load_model(checkpoint_path)
    return model

def preprocess_input(sequence):
    """预处理输入序列"""
    AA = ['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 'E', 'D', 'K', 'R']
    max_length = 8
    
    # 处理序列长度
    processed_seq = sequence.ljust(max_length, 'I')[:max_length]
    
    # 生成one-hot编码
    encoding = []
    for aa in processed_seq:
        code = [1 if aa == aa_type else 0 for aa_type in AA]
        encoding.append(code)
    
    return np.array(encoding, dtype=np.float32).reshape(1, max_length, 20)

def predict(model, sequence):
    """执行预测"""
    input_data = preprocess_input(sequence)
    prediction = model.predict(input_data)
    return prediction[0][0]

if __name__ == "__main__":
    # 配置参数
    CHECKPOINT_DIR = "./model/CNN_classification"  # checkpoint目录
    TEST_SEQUENCE = "KQTDAVKK"                   # 输入测试序列
    
    # 加载模型
    sess = load_checkpoint(CHECKPOINT_DIR)
    # 执行预测
    prob = predict(sess, TEST_SEQUENCE)
    print(f"Input Sequence: {TEST_SEQUENCE}")
    print(f"Predicted Probability: {prob:.4f}")