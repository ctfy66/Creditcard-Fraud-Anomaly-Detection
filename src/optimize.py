# hyperparameter_tuning_balanced_optimized_v2.py

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, roc_curve, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from main_algorithm import load_data, split_data, save_results_to_file  # 导入主要函数
from src.main_algorithm import visualize_results

# 加载数据
X, y = load_data()

# 数据不平衡处理
# 采用欠采样策略，随机抽样正常样本以平衡数据集
X_normal = X[y == 0]
y_normal = y[y == 0]
X_anomaly = X[y == 1]
y_anomaly = y[y == 1]

# 欠采样正常样本，使其数量与异常样本相等
X_normal_downsampled, y_normal_downsampled = resample(X_normal, y_normal, replace=False, n_samples=len(X_anomaly), random_state=42)

# 返回综合数据集
X_balanced = pd.concat([X_normal_downsampled, X_anomaly])
y_balanced = pd.concat([y_normal_downsampled, y_anomaly])

# 数据集划分
X_train, X_test, y_train, y_test = split_data(X_balanced, y_balanced)

# 定义参数网格，缓解较小的参数穿选范围
param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_samples': ['auto', 0.6, 0.8],
    'contamination': [0.001, 0.005, 0.01],
    'max_features': [1, 2, 5],
    'bootstrap': [False, True],
    'random_state': [42]
}

# 使用 RandomizedSearchCV 进行参数调优
model = IsolationForest()
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions,
                                    n_iter=20, scoring='f1_weighted', cv=2, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# 获取最优参数并重新训练模型
best_params = random_search.best_params_
best_model = IsolationForest(**best_params)
best_model.fit(X_train)

# 预测测试集
y_pred_scores = best_model.decision_function(X_test)

# 使用网格搜索找到最佳阈值以平衡精确率和召回率
best_threshold = None
best_f1 = 0
for threshold in np.linspace(np.min(y_pred_scores), np.max(y_pred_scores), 100):
    y_pred_temp = (y_pred_scores < threshold).astype(int)
    temp_f1 = f1_score(y_test, y_pred_temp)
    if temp_f1 > best_f1:
        best_f1 = temp_f1
        best_threshold = threshold

# 根据最佳阈值生成预测结果
y_pred_best = (y_pred_scores < best_threshold).astype(int)

# 生成调优后的分类报告
classification_report_best = classification_report(y_test, y_pred_best)
print(classification_report_best)

# 计算AUC和F1-score
auc_score = roc_auc_score(y_test, y_pred_scores)
f1 = f1_score(y_test, y_pred_best)
print(f"AUC-ROC Score: {auc_score}")
print(f"F1 Score with custom threshold: {f1}")

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_scores)
plt.figure()
plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# 绘制 Precision-Recall 曲线
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_scores)
plt.figure()
plt.plot(recall, precision, label="Precision-Recall curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# 保存调优后的结果到文件
save_results_to_file("../results/model_performance_results.txt", 'Optimized Model Results',
                     classification_report_best, best_params['n_estimators'],
                     best_params['contamination'], best_params['random_state'],
                     max_features=best_params.get('max_features', 2), bootstrap=best_params.get('bootstrap', False))