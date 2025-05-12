import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import subprocess

# 加载数据集
data = loadmat('20Newsgroups.mat')
X = data['traindata']  # 文本特征矩阵
y = data['traingnd'].ravel()  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 保存训练数据到 LIBLINEAR 格式
def save_liblinear_format(X, y, filename):
    with open(filename, 'w') as f:
        if hasattr(X, "tocoo"):  # 检查是否是稀疏矩阵
            X = X.tocsr()  # 转换为 CSR 格式以便高效迭代
            for i in range(X.shape[0]):
                row_start = X.indptr[i]
                row_end = X.indptr[i + 1]
                row_data = [
                    f"{X.indices[j] + 1}:{X.data[j]}" for j in range(row_start, row_end)
                ]
                line = f"{int(y[i])} " + " ".join(row_data)
                f.write(line + '\n')
        else:  # 如果是密集矩阵
            for i in range(X.shape[0]):
                line = f"{int(y[i])} " + " ".join(
                    [f"{j + 1}:{X[i, j]}" for j in range(X.shape[1]) if X[i, j] != 0]
                )
                f.write(line + '\n')

train_file = 'train_data.txt'
save_liblinear_format(X_train, y_train, train_file)

# 选择训练方法
def train_model(train_file, model_file, solver_type):
    """
    使用 LIBLINEAR 训练模型
    :param train_file: 训练数据文件路径
    :param model_file: 输出模型文件路径
    :param solver_type: 训练方法类型（默认 1）
    """
    subprocess.run(['train', '-s', str(solver_type), train_file, model_file], check=True)

# 调用训练方法
model_file = 'model'
solver_type = 13 # 修改此值以选择不同的训练方法
train_model(train_file, model_file, solver_type)

# 加载模型权重作为特征
def load_model_weights(model_file):
    with open(model_file, 'r') as f:
        lines = f.readlines()
    weights = []
    if 'w\n' in lines:
        start_index = lines.index('w\n') + 1
        for line in lines[start_index:]:
            weights.extend([float(x) for x in line.split()])  # 拆分并转换为浮点数
    return np.array(weights)

weights = load_model_weights(model_file)

# 检查 weights 的形状
print(f"X_train shape: {X_train.shape}")
print(f"weights shape: {weights.shape}")

# 如果 weights 是一维数组，调整其形状
if weights.ndim == 1:
    weights = weights[:X_train.shape[1]]  # 截取匹配的列数
    weights = weights.reshape(-1, 1)  # 转换为二维数组

# 对特征进行加权
X_train_weighted = X_train @ weights
X_test_weighted = X_test @ weights

# 标准化特征
scaler = StandardScaler(with_mean=False)  # 稀疏矩阵不能使用 with_mean=True
X_train_scaled = scaler.fit_transform(X_train_weighted)
X_test_scaled = scaler.transform(X_test_weighted)

# 使用KNN分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 测试集预测
y_pred = knn.predict(X_test_scaled)

# 输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"分类准确率: {accuracy * 100:.2f}%")