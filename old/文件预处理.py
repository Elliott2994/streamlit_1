from protlearn.features import (
    paac,
    aac,
    aaindex1,
    entropy,
    atc,
    ctd,
    moran,
    geary,
    qso,
    ngram,
    motif,
    cksaap,
)
from protlearn.preprocessing import remove_unnatural
from protlearn.dimreduction import tree_importance
import pandas as pd
import numpy as np
from icecream import ic

# sklean
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


# 测试集创建
train_data = pd.read_csv("../train_data/trainamp.csv", header=None)
# ic(train_data.head)
# # 提取aa_seq列为amptrain
amptrain = train_data.iloc[:, 1]
isAMP = train_data.iloc[:, 3]
# # 变成1D数组
isAMP = np.array(isAMP)
# # amptrain转成list
amptrain = np.array(amptrain)
amptrain = amptrain.tolist()
# # 变换isAMP中第一列True为1，False为0
isAMP = np.where(isAMP == True, 1, 0)
isAMP = isAMP.tolist()
# # 保存isMAP为csv
# np.savetxt("isAMP.csv", isAMP, delimiter=",")

# ic(amptrain)
# ic(isAMP)

# aac_train, _ = aac(amptrain)
aaindex1_train, aaindex1_inds = aaindex1(amptrain)
ng_train, ng_inds = ngram(amptrain)
ctd_train, ctd_inds = ctd(amptrain)
atc_train, atc_inds = atc(amptrain)
# aaindex1_inds = pd.DataFrame(atc_inds)
# aaindex1_inds.to_csv("aaindex1_inds.csv", index=False, header=False)
# ic(aaindex1_train.shape, ng_train.shape, ctd_train.shape, atc_train.shape)
# aac_train_df = pd.DataFrame(aac_train)
# aac_train_df.to_csv("aac_train.csv", index=False, header=False)

# ctd_train, _ = ctd(amptrain)
# ctd_train_df = pd.DataFrame(ctd_train)
# ctd_train_df.to_csv("ctd_train.csv", index=False, header=False)

# atoms_train, _ = atc(amptrain)
# atoms_train_df = pd.DataFrame(atoms_train)
# atoms_train_df.to_csv("atc_train.csv", index=False, header=False)

# 结果合并
train_fea = np.concatenate((aaindex1_train, ng_train, ctd_train, atc_train), axis=1)
ic(train_fea.shape)
# train_fea_df = pd.DataFrame(train_fea)
# train_fea_df.to_csv("train_fea.csv", index=False, header=False)

# treeimportance
reduced, indices = tree_importance(
    train_fea, isAMP, top=550, method="random_forest", max_depth=50
)
# ic(reduced.shape)
ic(indices)
# reduced_df = pd.DataFrame(reduced)
# reduced_df.to_csv("reduced_df.csv", index=False, header=False)
# indices_df = pd.DataFrame(indices)
# indices_df.to_csv("indices_df.csv", index=False, header=False)

# train_fea = train_fea[:, indices]
# ic(train_fea.shape)

#

# 验证集转换
# testdata = pd.read_csv("NonAMP_S3.fasta", header=None)
# test_data = testdata.iloc[1::2, 0]
# test_data = np.array(test_data)
# test_target = np.zeros(test_data.shape[0])
# test_data = test_data.tolist()
# # 输出csv文件，test_data为第一列，test_target为第二列
# pd.DataFrame({"test_data": test_data, "test_target": test_target}).to_csv(
#     "AMP_S3.csv", index=False
# )

# # 验证集读取
# testdata = pd.read_csv("../amp_test.csv", header=0)
# test_data = testdata.iloc[:, 0]
# test_data = np.array(test_data)
# test_data = test_data.tolist()
# test_target = testdata.iloc[:, 1]
# test_target = np.array(test_target)
# test_target = test_target.tolist()
# indices = pd.read_csv("../train_data/indices_df.csv", header=None)
# indices = indices.iloc[:, 0]
# indices = np.array(indices)
# indices = indices.tolist()
# ic(indices)


# 去除unnatural amino acid
# test_data = remove_unnatural(test_data)

# # 验证集预处理 old
# test_data, _ = aaindex1(test_data)
# test_data_df = pd.DataFrame(test_data)
# ic(test_data_df.shape)
# test_data_df.to_csv("test_data4.csv", index=False, header=False)

# 验证集预处理
# test_data_aaindex1, _ = aaindex1(test_data)
# test_data_ng, _ = ngram(test_data)
# test_data_ctd, _ = ctd(test_data)
# test_data_atc, _ = atc(test_data)
# test_data = np.concatenate(
#     (test_data_aaindex1, test_data_ng, test_data_ctd, test_data_atc), axis=1
# )
# ic(test_data.shape)
# test_data = test_data[:, indices]
# ic(test_data.shape)


# # 训练集读取
# train_data = pd.read_csv("../train_data/reduced_df.csv", header=None)
# train_data = np.array(train_data)
# train_target = pd.read_csv("../train_data/isAMP.csv", header=None)
# train_target = train_target.iloc[:, 0]
# train_target = np.array(train_target)
# ic(train_data.shape)
# ic(train_target.shape)

# rf = RandomForestClassifier(
#     n_estimators=203, max_depth=11, max_features=0.1, min_samples_split=24
# )

# 训练方法
# rf = RandomForestClassifier()
# rf.fit(train_data, train_target)

# gbdt = GradientBoostingClassifier()
# gbdt.fit(train_data, train_target)

# lr = LogisticRegression()
# lr.fit(train_data, train_target)

# knn = KNeighborsClassifier()
# knn.fit(train_data, train_target)

# svm = SVC()
# svm.fit(train_data, train_target)

# ada = AdaBoostClassifier()
# ada.fit(train_data, train_target)


# ic(rf.score(test_data, test_target))
# ic(gbdt.score(test_data, test_target))
# ic(lr.score(test_data, test_target))
# ic(knn.score(test_data, test_target))
# ic(svm.score(test_data, test_target))
# ic(ada.score(test_data, test_target))

# ic(pred)
# pred输出csv
# np.savetxt("pred.csv", pred, delimiter=",")
# 输出f1score
# ic(f1_score(test_target, pred))
# ic(matthews_corrcoef(test_target, pred))
# ic(roc_auc_score(test_target, pred))

# 集成学习 随机森林+gbdt
# rf = RandomForestClassifier()
# gbdt = GradientBoostingClassifier()
# rf_gbdt = VotingClassifier(estimators=[("rf", rf), ("gbdt", gbdt)], voting="soft")
# rf_gbdt.fit(train_data, train_target)
# pred = rf_gbdt.predict(test_data)
# ic(accuracy_score(test_target, pred))
