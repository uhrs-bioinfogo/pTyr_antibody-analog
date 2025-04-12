import pandas as pd
import numpy as np
from tensorflow.keras import layers, Sequential, losses, optimizers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tqdm import tqdm

# 配置GPU内存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def build_network():
    # 先创建包含多网络层的列表
    conv_layers = [

        layers.Conv1D(filters=128, kernel_size=1, padding='same', activation=tf.nn.relu,input_shape=( 8, 20)),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=9, padding='same', activation=tf.nn.relu),
        layers.MaxPooling1D(2, data_format="channels_first"),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu),
        layers.MaxPooling1D(pool_size=2, strides=1),
        layers.Dropout(0.5),
    ]

    fc_layers = [
        layers.Dense(64, activation=tf.nn.relu),
        layers.MaxPooling1D(2),
        layers.Dense(32, activation=tf.nn.relu),
        layers.MaxPooling1D(2),
        layers.Dense(8, activation=tf.nn.relu),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation=tf.nn.tanh)
    ]
    conv_layers.extend(fc_layers)
    network = Sequential(conv_layers)
    network.build(input_shape=[None, 8, 20])
    network.compile(optimizer=optimizers.Adam(), loss='mean_squared_error', metrics=['mae'])
    network.summary()
    return network

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
    CHECKPOINT_DIR = "./model/CNN_classification/"  # checkpoint目录
    df = pd.read_csv("data/seqs.csv", encoding='utf-8')
    seqs = df["sequence"].to_list()
    
    # 加载模型
    model = load_checkpoint(CHECKPOINT_DIR)  # 变量名改为model更清晰
    
    # 初始化结果列表
    results = []
    
    # 遍历序列并进行预测
    for seq in tqdm(seqs):
        prob = predict(model, seq)  # 使用模型进行预测
        loged = np.log10(prob)
        
        # 收集结果
        results.append({
            'sequence': seq,
            'prob': prob,
            'loged': loged
        })
    
    # 转换为DataFrame并保存CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv("prediction_results.csv", index=False)
