# main_algorithm.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, roc_curve, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
#from main_algorithm import load_data, split_data, save_results_to_file  #

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 数据加载函数
def load_data(file_path='../data/card_transdata.csv'):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['fraud'])  # 输入特征
    y = df['fraud']  # 目标变量
    return X, y

# 数据集划分函数
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 初始化并训练模型函数
def init_and_train_model(X_train, n_estimators=70, contamination=0.01, random_state=42, max_features=2, bootstrap=False):
    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state,
                          max_features=max_features, bootstrap=bootstrap)
    clf.fit(X_train)
    return clf

# 模型预测函数
def predict(clf, X_test):
    y_pred = clf.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred]  # 1表示欺诈，0表示正常
    return y_pred

# 评估结果保存函数
def save_results_to_file(file_path, description, classification_report_str, n_estimators, contamination, random_state, max_features, bootstrap):
    with open(file_path, 'a') as file:  # 使用 'a' 模式追加内容到文件
        file.write("Model parameters:\n")
        file.write(f"n_estimators = {n_estimators}, contamination = {contamination}, random_state = {random_state}, max_features = {max_features}, bootstrap = {bootstrap}\n")
        file.write(f"--- {description} ---\n")
        file.write("Classification Report:\n")
        file.write(classification_report_str + "\n")
        file.write("\n\n")  # 添加空行以便区分不同实验结果

# 可视化函数
def visualize_results(X_test, y_pred, save_path=None):
    X_test_values = X_test.iloc[:, [0, 1]].values  # 使用前两个特征进行可视化
    plt.scatter(X_test_values[:, 0], X_test_values[:, 1], c=y_pred, cmap='coolwarm')
    plt.title('Isolation Forest Anomaly Detection (Fraud Detection)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Fraud (1) vs Normal (0)')
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)  # 以高分辨率保存图像
        print(f"Visualization saved as {save_path}")
    plt.show()

# 主函数，用于训练初始模型并保存结果
def main():
    # 加载数据
    X, y = load_data()
    X_normal = X[y == 0]
    y_normal = y[y == 0]
    X_anomaly = X[y == 1]
    y_anomaly = y[y == 1]

    # 欠采样正常样本，使其数量与异常样本相等
    X_normal_downsampled, y_normal_downsampled = resample(X_normal, y_normal, replace=False, n_samples=len(X_anomaly),
                                                          random_state=42)

    # 返回综合数据集
    X_balanced = pd.concat([X_normal_downsampled, X_anomaly])
    y_balanced = pd.concat([y_normal_downsampled, y_anomaly])

    # 数据集划分
    X_train, X_test, y_train, y_test = split_data(X_balanced, y_balanced)

    # 初始化并训练模型
    clf = init_and_train_model(X_train)

    # 模型预测
    y_pred = predict(clf, X_test)

    # 评估结果
    classification_report_str = classification_report(y_test, y_pred)

    # 保存实验结果到文件
    save_results_to_file("../results/model_performance_results.txt", 'Initial Model Results',
                         classification_report_str, n_estimators=100, contamination=0.005,
                         random_state=42, max_features=2, bootstrap=False)

    # 可视化结果
    visualize_results(X_test, y_pred, save_path="../results/init_isolation_forest_result.png")

if __name__ == "__main__":
    main()
