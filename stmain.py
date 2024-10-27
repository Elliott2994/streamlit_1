import streamlit as st
import streamlit.components.v1 as html
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


def _feaprocess(data):
    # test_data, _ = aaindex1(data)
    # test_data_df = pd.DataFrame(test_data)
    test_data_aaindex1, _ = aaindex1(data)
    test_data_ng, _ = ngram(data)
    test_data_ctd, _ = ctd(data)
    test_data_atc, _ = atc(data)
    test_data = np.concatenate(
        (test_data_aaindex1, test_data_ng, test_data_ctd, test_data_atc), axis=1
    )
    indices = pd.read_csv("C:\\Users\\86151\\Desktop\\基于机器学习的抗菌肽预测模型\\代码和数据\\train_data\\indices_df.csv", header=None)
    indices = indices.iloc[:, 0]
    indices = np.array(indices)
    indices = indices.tolist()
    test_data = test_data[:, indices]

    train_data = pd.read_csv("C:\\Users\\86151\\Desktop\\基于机器学习的抗菌肽预测模型\\代码和数据\\train_data\\reduced_df.csv", header=None)
    train_data = np.array(train_data)
    train_target = pd.read_csv("C:\\Users\\86151\\Desktop\\基于机器学习的抗菌肽预测模型\\代码和数据\\train_data\\isAMP.csv", header=None)
    train_target = train_target.iloc[:, 0]
    train_target = np.array(train_target)

    return test_data, train_data, train_target


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


# st.set_page_config(page_title="抗菌肽预测")
st.set_page_config(
    page_title="抗菌肽预测模型",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": "# 基于计算方法的抗菌肽预测模型",
    },
)

colindex = 0

# st.balloons()
# st.title("抗菌肽（AMP）预测")
# st.markdown("### 1.数据预处理")
centered_text = "<center><h1>🧬 基于计算方法的抗菌肽预测模型<h1></center>"
st.markdown(centered_text, unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["开始预测", "关于模型", "联系我们"])

#

#
with tab2:
    st.subheader("🔬 关于模型")
    st.markdown("基于计算方法的抗菌肽预测模型——*作者：顾心愉*")
    st.subheader("")
    st.subheader("✂️ 特征处理方法")
    st.markdown(
        "本模型使用**aaindex1、ngram、ctd、atc**四种特征表示方法将蛋白质序列处理成计算机可识别的形式，并按照特征重要性排序并选取前50%的特征作为最终特征向量，以加快模型预测速度"
    )
    st.subheader("")
    st.subheader("📈 预测模型")
    st.markdown(
        "使用者可自由选用我们已经训练好的**随机森林算法RF**或**梯度提升算法GBDT**作为预测抗菌肽的模型，经过我们实验验证，这两种算法都具有较高的精度（ACC、F1 score、ROC_AUC)和马修相关系数(MCC)"
    )
with tab3:
    st.subheader("☎️ 联系我们")
    st.markdown("学术机构:徐州医科大学")
    st.markdown("作者:顾心愉")
    st.markdown("作者邮箱:2253608490@qq.com")


with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📤 上传肽序列数据")
        # st.write("- 上传需要检测的肽数据(**CSV**格式,每行为一条肽序列)")
        test_file = st.file_uploader(
            "上传需要预测的肽数据(**CSV**格式,每行为一条肽序列)", type=["csv"]
        )
        if test_file is not None:
            colindex = 1
            testdata = pd.read_csv(test_file, header=0)
            test_data = testdata.iloc[:, 0]
            test_data = np.array(test_data)
            test_data = test_data.tolist()
            st.write("肽序列数据预览,文件中共有" + str(len(test_data)) + "条肽序列")
            # 展示testdata第一列的数据预览
            st.write(testdata.iloc[:, 0])
        with col2:
            if colindex == 1:
                st.subheader("🛠️ 选择预测模型")
                model = st.radio(
                    label="选择一种预测模型",
                    options=(
                        "随机森林 RF",
                        "梯度提升 Gradient Boosting Classifier",
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
                # 按钮，开始预测
                if st.button("开始预测"):
                    # 显示旋转加载器
                    with st.spinner("预测中..."):
                        test_data_df, train_data, train_target = _feaprocess(test_data)

                        if model == "随机森林 RF":
                            pred = _rf(train_data, train_target, test_data_df)
                        elif model == "梯度提升 Gradient Boosting Classifier":
                            pred = _gb(train_data, train_target, test_data_df)
                        colindex = 2
                        #

            with col3:
                if colindex == 2:
                    st.subheader("📃 预测结果")
                    # restxt = "<center><h3>预测结果<h3></center>"
                    # st.markdown(restxt, unsafe_allow_html=True)

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
                            "background-color: #88DFF2"
                            if x == "是"
                            else "background-color: white"
                        )
                    )
                    st.write(
                        f"可能是抗菌肽（AMP）的序列共{x}条，占比{round(y*100,2)}%。"
                    )
                    st.write(
                        f"可能不是肽（NonAMP）的序列共{len(testdata)-x}条，占比{round(100-round(y*100,2),2)}%。"
                    )
                    st.write(result)
                    # 重新预测按钮，红色
                    if st.button("重新预测"):
                        st.experimental_rerun()
