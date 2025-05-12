from liblinearutil import *

# 读取训练数据
train_labels, train_features = svm_read_problem('heart_scale')

# 训练模型
model = train(train_labels, train_features, '-s 0 -c 1')

# 保存模型
save_model('model_file', model)

# 读取测试数据
test_labels, test_features = svm_read_problem('test_data.txt')

# 加载模型
model = load_model('model_file')

# 使用模型进行预测
predicted_labels, accuracy, decision_values = predict(test_labels, test_features, model)

# 输出预测结果
print("预测标签:", predicted_labels)
print("准确率:", accuracy)
print("决策值:", decision_values)