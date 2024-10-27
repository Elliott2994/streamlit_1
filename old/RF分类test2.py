from protlearn.preprocessing import onehot_encode
from protlearn.features import paac, aac, aaindex1, entropy, atc, ctd, moran, geary, qso
from protlearn.dimreduction import pca, tree_importance
import numpy as np
import pandas as pd
from icecream import ic

# sklean
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

# # # AMP,917 ampnon,1461 paac20,aac20,aaindex1 553

# 生成分类标签isAMP，前917为1，后1461为0
# isAMP = np.concatenate((np.ones(917), np.zeros(1461)))

# # 读取trainamp.csv
# train_data = pd.read_csv("trainamp.csv", header=None)
# # ic(train_data.head)
# # 提取aa_seq列为amptrain
# amptrain = train_data.iloc[:, 1]
# isAMP = train_data.iloc[:, 3]
# # 变成1D数组
# isAMP = np.array(isAMP)
# # amptrain转成list
# amptrain = np.array(amptrain)
# amptrain = amptrain.tolist()
# # 变换isAMP中第一列True为1，False为0
# isAMP = np.where(isAMP == True, 1, 0)
# # 保存isMAP为csv
# np.savetxt("isAMP.csv", isAMP, delimiter=",")

# ic(amptrain)
# ic(isAMP)

# aaindex1_amp, aaindex1_amp_desc = aaindex1(amptrain)
# aaindex1_feature_df = pd.DataFrame(aaindex1_amp)
# ic(aaindex1_amp.shape)
# aaindex1_feature_df.to_csv("aaindex1_amp.csv", index=False, header=False)

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

# 读取amp_s3
# amp_s3 = pd.read_csv("AMP_S3.csv", header=0)
# amp_s3_data = amp_s3.iloc[:, 0]
# amp_s3_data = np.array(amp_s3_data)
# amp_s3_data = amp_s3_data.tolist()
# amp_s3_target = amp_s3.iloc[:, 1]
# amp_s3_target = np.array(amp_s3_target)
# amp_s3_target = amp_s3_target.tolist()
# ic(amp_s3_data)
# ic(amp_s3_target)

# 读取NonAMP_S1.fa
# testdata = pd.read_csv("AMP_S1.fasta", header=None)
# test_data = testdata.iloc[1::2, 0]
# test_data = np.array(test_data)
# test_target = np.ones(test_data.shape[0])
# test_data = test_data.tolist()

# aaindex1_test_data, aaindex1_test_desc = aaindex1(test_data)
# aaindex1_test_df = pd.DataFrame(aaindex1_test_data)
# ic(aaindex1_test_df.shape)
# ic(test_target.shape)
#
# 特征提取与数据合并_AMP_AMPnon
# aac
# aac_amp, aac_amp_desc = aac("AMP.fasta")
# aac_ampnon, aac_ampnon_desc = aac("AMPnon.fasta")
# aac_feature = np.concatenate((aac_amp, aac_ampnon), axis=0)
# aac_feature_df = pd.DataFrame(aac_feature)
# ic(aac_amp.shape)

# # aaindex1
# aaindex1_amp, aaindex1_amp_desc = aaindex1("AMP.fasta")
# aaindex1_ampnon, aaindex1_ampnon_desc = aaindex1("AMPnon.fasta")
# aaindex1_feature = np.concatenate((aaindex1_amp, aaindex1_ampnon), axis=0)
# aaindex1_feature_df = pd.DataFrame(aaindex1_feature)
# ic(aaindex1_amp.shape)
# # entropy
# entropy_amp = entropy("AMP.fasta")
# entropy_ampnon = entropy("AMPnon.fasta")
# entropy_feature = np.concatenate((entropy_amp, entropy_ampnon), axis=0)
# entropy_feature_df = pd.DataFrame(entropy_feature)

# # atc
# atoms_amp, bonds_amp = atc("AMP.fasta")
# atoms_ampnon, bonds_ampnon = atc("AMPnon.fasta")
# atoms_feature = np.concatenate((atoms_amp, atoms_ampnon), axis=0)
# bonds_feature = np.concatenate((bonds_amp, bonds_ampnon), axis=0)
# atoms_feature_df = pd.DataFrame(atoms_feature)
# bonds_feature_df = pd.DataFrame(bonds_feature)

# # ctd
# ctd_amp, crd_amp_desc = ctd("AMP.fasta")
# ctd_ampnon, crd_ampnon_desc = ctd("AMPnon.fasta")
# ctd_feature = np.concatenate((ctd_amp, ctd_ampnon), axis=0)
# ctd_feature_df = pd.DataFrame(ctd_feature)

# # moran
# moran_amp = moran("AMP.fasta")
# moran_ampnon = moran("AMPnon.fasta")
# moran_feature = np.concatenate((moran_amp, moran_ampnon), axis=0)
# moran_feature_df = pd.DataFrame(moran_feature)

# # geary
# geary_amp = geary("AMP.fasta")
# geary_ampnon = geary("AMPnon.fasta")
# geary_feature = np.concatenate((geary_amp, geary_ampnon), axis=0)
# geary_feature_df = pd.DataFrame(geary_feature)

# # qso
# sw_amp, g_amp, qso_amp_desc = qso("AMP.fasta", d=0)
# sw_ampnon, g_ampnon, qso_ampnon_desc = qso("AMPnon.fasta", d=0)
# sw_feature = np.concatenate((sw_amp, sw_ampnon), axis=0)
# sw_feature_df = pd.DataFrame(sw_feature)
# #

#
# fourfeature = np.concatenate(
#     (atoms_feature, aac_feature, ctd_feature, sw_feature), axis=1
# )
# fivefeature = np.concatenate(
#     (aac_feature, atoms_feature, ctd_feature, sw_feature, aaindex1_feature), axis=1
# )
# fourfeature_df = pd.DataFrame(fourfeature)
# fivefeature_df = pd.DataFrame(fivefeature)
#

#
# ic(aac_feature_df.shape)
# ic(aaindex1_feature_df.shape)
# ic(entropy_feature_df.shape)
# ic(atoms_feature_df.shape)
# ic(bonds_feature_df.shape)
# ic(ctd_feature_df.shape)
# ic(moran_feature_df.shape)
# ic(geary_feature_df.shape)
# ic(sw_feature_df.shape)
# ic(fourfeature_df.shape)
# ic(fivefeature_df.shape)
#

# 数据切分
# aac_X_train, aac_X_test, aac_y_train, aac_y_test = train_test_split(
#     aac_feature_df, isAMP, test_size=0.3, random_state=42
# )
aaindex1_X_train, aaindex1_X_test, aaindex1_y_train, aaindex1_y_test = train_test_split(
    aaindex1_feature_df, isAMP, test_size=0.3, random_state=42
)
# entropy_X_train, entropy_X_test, entropy_y_train, entropy_y_test = train_test_split(
#     entropy_feature_df, isAMP, test_size=0.3, random_state=42
# )
# # atoms和bonds分开计算
# atoms_X_train, atoms_X_test, atoms_y_train, atoms_y_test = train_test_split(
#     atoms_feature_df, isAMP, test_size=0.3, random_state=42
# )
# bonds_X_train, bonds_X_test, bonds_y_train, bonds_y_test = train_test_split(
#     bonds_feature_df, isAMP, test_size=0.3, random_state=42
# )
# #
# ctd_X_train, ctd_X_test, ctd_y_train, ctd_y_test = train_test_split(
#     ctd_feature_df, isAMP, test_size=0.3, random_state=42
# )
# moran_X_train, moran_X_test, moran_y_train, moran_y_test = train_test_split(
#     moran_feature_df, isAMP, test_size=0.3, random_state=42
# )
# geary_X_train, geary_X_test, geary_y_train, geary_y_test = train_test_split(
#     geary_feature_df, isAMP, test_size=0.3, random_state=42
# )
# sw_X_train, sw_X_test, sw_y_train, sw_y_test = train_test_split(
#     sw_feature_df, isAMP, test_size=0.3, random_state=42
# )
# fourfeature_X_train, fourfeature_X_test, fourfeature_y_train, fourfeature_y_test = (
#     train_test_split(fourfeature_df, isAMP, test_size=0.3, random_state=42)
# )
# fivefeature_X_train, fivefeature_X_test, fivefeature_y_train, fivefeature_y_test = (
#     train_test_split(fivefeature_df, isAMP, test_size=0.3, random_state=42)
# )

# #

# 模型
rf = RandomForestClassifier(random_state=42)

# # 模型拟合与预测
# rf.fit(aac_X_train, aac_y_train)
# aac_y_pred = rf.predict(aac_X_test)

rf.fit(aaindex1_X_train, aaindex1_y_train)
aaindex1_y_pred = rf.predict(aaindex1_X_test)

# rf.fit(entropy_X_train, entropy_y_train)
# entropy_y_pred = rf.predict(entropy_X_test)

# rf.fit(atoms_X_train, atoms_y_train)
# atoms_y_pred = rf.predict(atoms_X_test)

# rf.fit(bonds_X_train, bonds_y_train)
# bonds_y_pred = rf.predict(bonds_X_test)

# rf.fit(ctd_X_train, ctd_y_train)
# ctd_y_pred = rf.predict(ctd_X_test)

# rf.fit(moran_X_train, moran_y_train)
# moran_y_pred = rf.predict(moran_X_test)

# rf.fit(geary_X_train, geary_y_train)
# geary_y_pred = rf.predict(geary_X_test)

# rf.fit(sw_X_train, sw_y_train)
# sw_y_pred = rf.predict(sw_X_test)

# rf.fit(fourfeature_X_train, fourfeature_y_train)
# fourfeature_y_pred = rf.predict(fourfeature_X_test)

# rf.fit(fivefeature_X_train, fivefeature_y_train)
# fivefeature_y_pred = rf.predict(fivefeature_X_test)

# rf.fit(aaindex1_feature_df, isAMP)
# pred = rf.predict(aaindex1_test_df)
# ic(accuracy_score(test_target, pred))

# pred = rf.predict(aaindex1_nonamps1_data)
# ic(accuracy_score(nonamp_s1_target, pred))

# # 结果输出
# ic(accuracy_score(aac_y_test, aac_y_pred))
ic(accuracy_score(aaindex1_y_test, aaindex1_y_pred))
# ic(accuracy_score(entropy_y_test, entropy_y_pred))
# ic(accuracy_score(atoms_y_test, atoms_y_pred))
# ic(accuracy_score(bonds_y_test, bonds_y_pred))
# ic(accuracy_score(ctd_y_test, ctd_y_pred))
# ic(accuracy_score(moran_y_test, moran_y_pred))
# ic(accuracy_score(geary_y_test, geary_y_pred))
# ic(accuracy_score(sw_y_test, sw_y_pred))
# ic(accuracy_score(fourfeature_y_test, fourfeature_y_pred))
# ic(accuracy_score(fivefeature_y_test, fivefeature_y_pred))

# ic(classification_report(aaindex1_y_test, aaindex1_y_pred))
ic(f1_score(aaindex1_y_test, aaindex1_y_pred, average="macro"))
ic(roc_auc_score(aaindex1_y_test, aaindex1_y_pred, average="macro"))
ic(matthews_corrcoef(aaindex1_y_test, aaindex1_y_pred))
