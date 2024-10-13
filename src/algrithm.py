import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载CSV数据
df = pd.read_csv('ai_basic/data/card_transdata.csv')  # 确保路径正确
# 分离目标变量和输入特征
X = df.drop(columns=['fraud'])  # 输入特征
y = df['fraud']  # 目标变量

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化并训练孤独森林模型
clf = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)
clf.fit(X_train)

# 对测试集进行预测
y_pred = clf.predict(X_test)

# 将预测结果中的1（正常）和-1（异常）调整为0和1，以便与目标变量对比
y_pred = [1 if x == -1 else 0 for x in y_pred]  # 1表示欺诈，0表示正常

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 输出详细的分类报告
print(classification_report(y_test, y_pred))

# 可视化，假设我们使用前两个特征
X_test_values = X_test.iloc[:, [0,1]].values  # 使用前两个特征进行可视化

# 将预测结果的类别映射为颜色，使用c参数传递数值数组
# 1 表示红色 (欺诈)，0 表示蓝色 (正常)
plt.scatter(X_test_values[:, 0], X_test_values[:, 1], c=y_pred, cmap='coolwarm')
plt.title('Isolation Forest Anomaly Detection (Fraud Detection)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Fraud (1) vs Normal (0)')
plt.show()
