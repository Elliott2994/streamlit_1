import streamlit as st
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
import pandas as pd
import numpy as np

# from icecream import ic

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


def _aaindex1(data):
    test_data, _ = aaindex1(data)
    test_data_df = pd.DataFrame(test_data)
    train_data = pd.read_csv("./train_data/aaindex1_train.csv", header=None)
    train_data = np.array(train_data)
    train_target = pd.read_csv("./train_data/isAMP.csv", header=None)
    train_target = train_target.iloc[:, 0]
    train_target = np.array(train_target)
    return test_data_df, train_data, train_target


def _aac(data):
    test_data, _ = aac(data)
    test_data_df = pd.DataFrame(test_data)
    train_data = pd.read_csv("./train_data/aac_train.csv", header=None)
    train_data = np.array(train_data)
    train_target = pd.read_csv("./train_data/isAMP.csv", header=None)
    train_target = train_target.iloc[:, 0]
    train_target = np.array(train_target)
    return test_data_df, train_data, train_target


def _atc(data):
    test_data, _ = atc(data)
    test_data_df = pd.DataFrame(test_data)
    train_data = pd.read_csv("./train_data/atc_train.csv", header=None)
    train_data = np.array(train_data)
    train_target = pd.read_csv("./train_data/isAMP.csv", header=None)
    train_target = train_target.iloc[:, 0]
    train_target = np.array(train_target)
    return test_data_df, train_data, train_target


def _ctd(data):
    test_data, _ = ctd(data)
    test_data_df = pd.DataFrame(test_data)
    train_data = pd.read_csv("./train_data/ctd_train.csv", header=None)
    train_data = np.array(train_data)
    train_target = pd.read_csv("./train_data/isAMP.csv", header=None)
    train_target = train_target.iloc[:, 0]
    train_target = np.array(train_target)
    return test_data_df, train_data, train_target


def _rf(train_data, train_target, test_data):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(train_data, train_target)
    pred = rf.predict(test_data)
    return pred


def _gb(train_data, train_target, test_data):
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(train_data, train_target)
    pred = gb.predict(test_data)
    return pred


def _lr(train_data, train_target, test_data):
    lr = LogisticRegression(random_state=42)
    lr.fit(train_data, train_target)
    pred = lr.predict(test_data)
    return pred


def _knn(train_data, train_target, test_data):
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_target)
    pred = knn.predict(test_data)
    return pred


def _svm(train_data, train_target, test_data):
    svm = SVC(random_state=42)
    svm.fit(train_data, train_target)
    pred = svm.predict(test_data)
    return pred


def _ada(train_data, train_target, test_data):
    ada = AdaBoostClassifier(random_state=42)
    ada.fit(train_data, train_target)
    pred = ada.predict(test_data)
    return pred


st.set_page_config(page_title="抗菌肽预测")
st.title("抗菌肽（AMP）预测")
# st.markdown("### 1.数据预处理")
st.subheader("Step 1 上传肽序列数据")
# st.write("- 上传需要检测的肽数据(**CSV**格式,每行为一条肽序列)")
test_file = st.file_uploader(
    "上传需要预测的肽数据(**CSV**格式,每行为一条肽序列)", type=["csv"]
)
if test_file is not None:
    testdata = pd.read_csv(test_file, header=0)
    test_data = testdata.iloc[:, 0]
    test_data = np.array(test_data)
    test_data = test_data.tolist()
    st.write("肽序列数据预览,共有" + str(len(test_data)) + "条肽序列")
    # 展示testdata第一列的数据预览
    st.write(testdata.iloc[:, 0])
    st.subheader("Step 2 选择肽序列的特征提取方法")
    # 单选，选择aaindex1，aac，atc，ctd这几个选项
    method = st.selectbox(
        label="点击下拉框，选择一种特征提取方法",
        options=("aaindex1", "aac", "atc", "ctd"),
        index=0,
    )
    if method == "aaindex1":
        st.write(
            "- **方法说明**：这种特征提取方法,基于AAIndex1的物理化学性质。包含566个索引,其中553个不包含NaN。通过收集序列中每个氨基酸的指数,然后在整个序列中取平均值"
        )
    elif method == "aac":
        st.write(
            "- **方法说明**：aac,即Amino acid composition. 氨基酸组成。这种特征提取方法计算每个序列的氨基酸频率"
        )
    elif method == "atc":
        st.write(
            "- **方法说明**：atc方法计算每个氨基酸序列的原子和键组成的总和。原子特征由五个原子(C、H、N、O和S)组成,并且键特征由总键、单键和双键组成。"
        )
    elif method == "ctd":
        st.write(
            "- **方法说明**：ctd,联合三和弦描述符,氨基酸可以根据它们的偶极和侧链体积分为7个不同的类别,这反映了它们的静电和疏水相互作用。"
        )
    st.subheader("Step 3 选择预测模型")
    model = st.selectbox(
        label="点击下拉框，选择一种预测模型",
        options=(
            "随机森林 RF",
            "梯度提升 Gradient Boosting Classifier",
            "逻辑回归 Logistic Regression",
            "k-近邻算法 KNN",
            "支持向量机 SVM",
            "自适应提升算法 AdaBoost",
        ),
        index=0,
    )
    if model == "随机森林 RF":
        st.write(
            "- **方法说明**：随机森林是一种集成学习方法，它通过构建多个决策树来预测目标变量。每个决策树都使用不同的特征和随机采样，然后通过投票或平均来预测目标变量。"
        )
    elif model == "梯度提升 Gradient Boosting Classifier":
        st.write(
            "- **方法说明**：梯度提升分类器（Gradient Boosting Classifier）是一种集成学习方法，基于梯度提升（Gradient Boosting）框架，主要用于解决分类问题。"
        )
    elif model == "逻辑回归 Logistic Regression":
        st.write(
            "- **方法说明**：逻辑回归是一种分类算法，它通过计算每个特征的权重来预测目标变量。"
        )
    elif model == "k-近邻算法 KNN":
        st.write(
            "- **方法说明**：KNN全称为K-Nearest Neighbors分类器（K-近邻算法），是一种基于实例的学习方法。"
        )
    elif model == "支持向量机 SVM":
        st.write(
            "- **方法说明**：支持向量机（Support Vector Machine, SVM）是一种广泛应用于分类和回归分析的监督学习模型。"
        )
    elif model == "自适应提升算法 AdaBoost":
        st.write(
            "- **方法说明**：AdaBoost通过将多个弱分类器组合成一个强分类器来提高分类性能。"
        )
    # 按钮，开始预测
    if st.button("开始预测"):
        # 显示旋转加载器
        with st.spinner("预测中..."):
            if method == "aaindex1":
                test_data_df, train_data, train_target = _aaindex1(test_data)
            elif method == "aac":
                test_data_df, train_data, train_target = _aac(test_data)
            elif method == "atc":
                test_data_df, train_data, train_target = _atc(test_data)
            elif method == "ctd":
                test_data_df, train_data, train_target = _ctd(test_data)

            if model == "随机森林 RF":
                pred = _rf(train_data, train_target, test_data_df)
            elif model == "梯度提升 Gradient Boosting Classifier":
                pred = _gb(train_data, train_target, test_data_df)
            elif model == "逻辑回归 Logistic Regression":
                pred = _lr(train_data, train_target, test_data_df)
            elif model == "k-近邻算法 KNN":
                pred = _knn(train_data, train_target, test_data_df)
            elif model == "支持向量机 SVM":
                pred = _svm(train_data, train_target, test_data_df)
            elif model == "自适应提升算法 AdaBoost":
                pred = _ada(train_data, train_target, test_data_df)
            #
        st.subheader("预测结果")

        # 将pred，1为是，0为否
        pred = pd.DataFrame(pred)
        x = pred.value_counts()[1]
        y = x / len(pred)
        pred = pred.replace(0, "否")
        pred = pred.replace(1, "是")
        # 构建一个dataframe,第一列为肽序列，数据为test_data,第二列为是否为AMP肽，数据为pred
        result = pd.DataFrame(test_data, columns=["肽序列"])
        result["是否为AMP肽"] = pred
        # 是否为AMP肽列，是的数据改变颜色
        result = result.style.map(
            lambda x: (
                "background-color: #88DFF2" if x == "是" else "background-color: white"
            )
        )
        st.write(f"可能是抗菌肽（AMP）的序列共{x}条，占比{round(y*100,2)}%。")
        st.write(
            f"可能不是肽（NonAMP）的序列共{len(testdata)-x}条，占比{round(100-round(y*100,2),2)}%。"
        )
        st.write(result)
        # 重新预测按钮，红色
        if st.button("重新预测"):
            st.experimental_rerun()
