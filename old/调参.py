from protlearn.preprocessing import onehot_encode
from protlearn.features import paac, aac, aaindex1, entropy, atc, ctd, moran, geary, qso
from protlearn.dimreduction import pca, tree_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic

# sklean
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# # # AMP,917 ampnon,1461 paac20,aac20,aaindex1 553


#
# 特征提取与数据合并_AMP_AMPnon

# 读取aaindex1_amp.csv
aaindex1_amp = pd.read_csv("aaindex1_train.csv", header=None)
aaindex1_feature_df = np.array(aaindex1_amp)
# # 读取isAMP.csv
isAMP = pd.read_csv("isAMP.csv", header=None)
# 提取isAMP第一列
isAMP = isAMP.iloc[:, 0]
isAMP = np.array(isAMP)
# ic(isAMP)
# ic(aaindex1_amp)
#

#

# 数据切分
aaindex1_X_train, aaindex1_X_test, aaindex1_y_train, aaindex1_y_test = train_test_split(
    aaindex1_feature_df, isAMP, test_size=0.3, random_state=42
)

#

# 模型
rf = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier()
lr = LogisticRegression(random_state=42)
svm = SVC(random_state=42)
ada = AdaBoostClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
# sklist = [rf, knn, lr, svm, ada, gb]

# 模型拟合与预测

# rf.fit(aaindex1_X_train, aaindex1_y_train)
# rf_aaindex1_y_pred = rf.predict(aaindex1_X_test)
# knn.fit(aaindex1_X_train, aaindex1_y_train)
# knn_aaindex1_y_pred = knn.predict(aaindex1_X_test)
# lr.fit(aaindex1_X_train, aaindex1_y_train)
# lr_aaindex1_y_pred = lr.predict(aaindex1_X_test)
# svm.fit(aaindex1_X_train, aaindex1_y_train)
# svm_aaindex1_y_pred = svm.predict(aaindex1_X_test)
ada.fit(aaindex1_X_train, aaindex1_y_train)
ada_aaindex1_y_pred = ada.predict(aaindex1_X_test)
gb.fit(aaindex1_X_train, aaindex1_y_train)
gb_aaindex1_y_pred = gb.predict(aaindex1_X_test)


# 结果输出
# ic("rf", accuracy_score(aaindex1_y_test, rf_aaindex1_y_pred))
# ic("matthews_corrcoef", matthews_corrcoef(aaindex1_y_test, rf_aaindex1_y_pred))
# ic("f1_score", f1_score(aaindex1_y_test, rf_aaindex1_y_pred))
# ic("roc_auc_score", roc_auc_score(aaindex1_y_test, rf_aaindex1_y_pred))

# ic("knn", accuracy_score(aaindex1_y_test, knn_aaindex1_y_pred))
# ic("matthews_corrcoef", matthews_corrcoef(aaindex1_y_test, knn_aaindex1_y_pred))
# ic("f1_score", f1_score(aaindex1_y_test, knn_aaindex1_y_pred))
# ic("roc_auc_score", roc_auc_score(aaindex1_y_test, knn_aaindex1_y_pred))

# ic("lr", accuracy_score(aaindex1_y_test, lr_aaindex1_y_pred))
# ic("matthews_corrcoef", matthews_corrcoef(aaindex1_y_test, lr_aaindex1_y_pred))
# ic("f1_score", f1_score(aaindex1_y_test, lr_aaindex1_y_pred))
# ic("roc_auc_score", roc_auc_score(aaindex1_y_test, lr_aaindex1_y_pred))

# ic("svm", accuracy_score(aaindex1_y_test, svm_aaindex1_y_pred))
# ic("matthews_corrcoef", matthews_corrcoef(aaindex1_y_test, svm_aaindex1_y_pred))
# ic("f1_score", f1_score(aaindex1_y_test, svm_aaindex1_y_pred))
# ic("roc_auc_score", roc_auc_score(aaindex1_y_test, svm_aaindex1_y_pred))

ic("ada", accuracy_score(aaindex1_y_test, ada_aaindex1_y_pred))
ic("matthews_corrcoef", matthews_corrcoef(aaindex1_y_test, ada_aaindex1_y_pred))
ic("f1_score", f1_score(aaindex1_y_test, ada_aaindex1_y_pred))
ic("roc_auc_score", roc_auc_score(aaindex1_y_test, ada_aaindex1_y_pred))

ic("gb", accuracy_score(aaindex1_y_test, gb_aaindex1_y_pred))
ic("matthews_corrcoef", matthews_corrcoef(aaindex1_y_test, gb_aaindex1_y_pred))
ic("f1_score", f1_score(aaindex1_y_test, gb_aaindex1_y_pred))
ic("roc_auc_score", roc_auc_score(aaindex1_y_test, gb_aaindex1_y_pred))


# 调参

# rf = RandomForestClassifier(
#     n_estimators=105,
#     max_depth=10,
#     class_weight="balanced",
# )
# rf = RandomForestClassifier()
# rf.fit(aaindex1_X_train, aaindex1_y_train)
# ic(rf.score(aaindex1_X_test, aaindex1_y_test))
# ic(cross_val_score(rf, aaindex1_X_test, aaindex1_y_test, cv=5).mean())

# 学习曲线绘制

# cross = []
# for i in range(0, 200, 10):
#     rf = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1, random_state=42)
#     cross_score = cross_val_score(rf, aaindex1_X_train, aaindex1_y_train, cv=5).mean()
#     cross.append(cross_score)
# plt.plot(range(1, 201, 10), cross)
# plt.xlabel("n_estimators")
# plt.ylabel("acc")
# plt.show()
# ic((cross.index(max(cross)) * 10) + 1, max(cross))

# 调整范围  171,,71
# cross = []
# for i in range(140, 165):
#     rf = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1, random_state=42)
#     cross_score = cross_val_score(rf, aaindex1_X_test, aaindex1_y_test, cv=5).mean()
#     cross.append(cross_score)
# plt.plot(range(140, 165), cross)
# plt.xlabel("n_estimators")
# plt.ylabel("acc")
# plt.show()
# ic(cross.index(max(cross)) + 1, max(cross))

# 深度
# param_grid = {"max_depth": np.arange(16, 20, 1)}
# param_grid = {"max_features": np.arange(13, 15, 1)}

# GS = GridSearchCV(rf, param_grid, cv=5)
# GS.fit(aaindex1_X_train, aaindex1_y_train)
# ic(GS.best_params_)
# ic(GS.best_score_)
