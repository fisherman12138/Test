import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import pandas as pd

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
    subprocess.run(['train', '-s', str(solver_type), train_file, model_file], check=True)

# 调用训练方法
model_file = 'model'
solver_type = 13  # 修改此值以选择不同的训练方法
train_model(train_file, model_file, solver_type)

# 加载模型权重作为特征
def load_model_weights(model_file, n_features):
    with open(model_file, 'r') as f:
        lines = f.readlines()
    weights = []
    if 'w\n' in lines:
        start_index = lines.index('w\n') + 1
        for line in lines[start_index:]:
            weights.extend([float(x) for x in line.split()])
    weights = np.array(weights)
    # 确保权重大小与特征数一致
    if weights.size > n_features:
        weights = weights[:n_features]
    return weights

# 加载权重
model_file = 'model'
weights = load_model_weights(model_file, X_train.shape[1])

# 检查权重形状
print(f"X_train shape: {X_train.shape}")
print(f"weights shape: {weights.shape}")

# 如果权重是一维数组，调整其形状
if weights.ndim == 1:
    weights = weights.reshape(-1, 1)

# 特征增强策略
def apply_feature_enhancement(X, weights, method='linear'):
    if method == 'linear':
        return X @ weights
    elif method == 'binary':
        threshold = np.median(np.abs(weights))
        binary_weights = np.where(np.abs(weights) > threshold, 1, 0)
        return X @ binary_weights
    elif method == 'nonlinear':
        signed_log_weights = np.sign(weights) * np.log1p(np.abs(weights))
        return X @ signed_log_weights
    else:
        raise ValueError("Invalid method")

# 对比不同特征增强方法
methods = ['linear', 'binary', 'nonlinear']
results = {}

for method in methods:
    X_train_weighted = apply_feature_enhancement(X_train, weights, method=method)
    X_test_weighted = apply_feature_enhancement(X_test, weights, method=method)

    # 标准化特征
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_weighted)
    X_test_scaled = scaler.transform(X_test_weighted)

    # 使用KNN分类器
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    results[method] = accuracy
    print(f"Method: {method}, Accuracy: {accuracy:.4f}")

# 可视化结果
plt.bar(results.keys(), results.values())
plt.ylabel('Accuracy')
plt.title('Feature Enhancement Methods Comparison')
plt.savefig('enhancement_comparison.png')

# 模型解释性分析
def analyze_feature_importance(weights, feature_names=None, top_n=10):
    abs_weights = np.abs(weights)
    top_indices = np.argsort(abs_weights)[-top_n:][::-1]

    plt.figure(figsize=(10, 6))
    if feature_names is not None:
        top_features = [feature_names[i] for i in top_indices]
        plt.barh(top_features, abs_weights[top_indices])
        plt.title(f'Top {top_n} Important Features')
    else:
        plt.hist(abs_weights, bins=50)
        plt.title('Distribution of Feature Weights')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

feature_names = None
analyze_feature_importance(weights, feature_names)

# 分类边界可视化
def visualize_class_boundary(X, y, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 'TSNE':
        reducer = TSNE(n_components=2, perplexity=30)
    else:
        raise ValueError("Invalid method")

    X_reduced = reducer.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab20', alpha=0.6)
    plt.title(f'{method} Visualization of Weighted Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter)
    plt.savefig(f'{method}_visualization.png')
    plt.show()

sample_idx = np.random.choice(X_train.shape[0], 1000, replace=False)
X_sample = X_train[sample_idx]
y_sample = y_train[sample_idx]

visualize_class_boundary(X_sample, y_sample, method='PCA')
visualize_class_boundary(X_sample, y_sample, method='TSNE')

# 噪声鲁棒性分析
def add_noise(X, noise_level=0.1):
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X.copy()
    noise = np.random.normal(0, noise_level, X_dense.shape)
    return X_dense + noise

noise_levels = [0.0, 0.05, 0.1, 0.2]
accuracies = []

for level in noise_levels:
    X_train_noisy = add_noise(X_train_scaled, level)
    X_test_noisy = add_noise(X_test_scaled, level)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_noisy, y_train)
    y_pred = knn.predict(X_test_noisy)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Noise Level: {level:.2f}, Accuracy: {acc:.4f}")

plt.plot(noise_levels, accuracies, marker='o')
plt.xlabel('Noise Level')
plt.ylabel('Accuracy')
plt.title('Model Robustness to Feature Noise')
plt.grid(True)
plt.savefig('noise_robustness.png')
plt.show()

# 参数扫描与批量实验
solver_types = [1, 5, 7, 13]
knn_neighbors = [3, 5, 7]

results = []
for solver in solver_types:
    train_model(train_file, model_file, solver_type=solver)
    weights = load_model_weights(model_file, X_train.shape[1])

    X_train_weighted = X_train @ weights.reshape(-1, 1)
    X_test_weighted = X_test @ weights.reshape(-1, 1)
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_weighted)
    X_test_scaled = scaler.transform(X_test_weighted)

    for k in knn_neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results.append({
            'solver_type': solver,
            'knn_neighbors': k,
            'accuracy': acc
        })

df_results = pd.DataFrame(results)
df_results.to_csv('hyperparameter_scan.csv', index=False)

pivot_table = df_results.pivot(index='solver_type', columns='knn_neighbors', values='accuracy')
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title('Hyperparameter Scan: Solver Type vs KNN Neighbors')
plt.xlabel('KNN Neighbors')
plt.ylabel('LIBLINEAR Solver Type')
plt.savefig('hyperparameter_heatmap.png')
plt.show()