import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 加载路径配置（需根据实际路径修改）
model_save_path = "model/CNN_regression_model/"  # 替换为实际模型路径
standard_file_path = "data/regression_standard.csv"  # 替换为标准数据文件路径

# 初始化标准化器（与训练时一致）
data = pd.read_csv(standard_file_path, sep=',')
standard_values = data['value'].values.reshape(-1, 1)

# 创建并拟合标准化器
scaler = StandardScaler().fit(standard_values)
minmax_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(scaler.transform(standard_values))

def one_hot_encode(sequence):
    """将8个氨基酸的序列编码为one-hot格式"""
    AA = ['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 
          'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 
          'E', 'D', 'K', 'R']
    
    # 检查序列长度
    if len(sequence) != 8:
        raise ValueError("Input sequence must be 8 amino acids long")
    
    encoding = []
    for aa in sequence:
        if aa == 'X':  # 处理未知氨基酸
            encoding += [0.05]*20
        else:
            encoding += [1 if aa == aa_class else 0 for aa_class in AA]
    return np.array(encoding).reshape(8, 20)

def predict_score(input_sequence):
    """预测氨基酸序列的活性分数"""
    # 加载模型
    model = tf.keras.models.load_model(model_save_path)
    
    # 数据预处理
    try:
        encoded = one_hot_encode(input_sequence)
    except ValueError as e:
        return str(e)
    
    # 格式转换
    input_tensor = tf.constant(encoded, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # 添加batch维度
    
    # 模型预测
    normalized_score = model.predict(input_tensor)[0][0]
    
    # 逆标准化处理
    score = minmax_scaler.inverse_transform([[normalized_score]])
    original_score = scaler.inverse_transform(score)
    
    return round(original_score[0][0], 4)

# 使用示例
if __name__ == "__main__":
    test_sequence = "LLLVAAPR"  # 替换为要预测的8肽序列
    try:
        score = predict_score(test_sequence)
        print(f"Predicted score for sequence {test_sequence}: {score}")
    except Exception as e:
        print(f"Error: {str(e)}")