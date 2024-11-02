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
import base64
from PIL import Image
from io import BytesIO

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
import joblib
import json  
import os  
 #加载模型 
def load_rf_model():  
    model_path = "rf_model.pkl"  
    model = joblib.load(model_path)  
    return model
def load_gb_model():  
    model_path = "gb_model.pkl"  
    model = joblib.load(model_path)  
    return model

def _feaprocess(data):#数据处理
    # test_data, _ = aaindex1(data)
    # test_data_df = pd.DataFrame(test_data)
    test_data_aaindex1, _ = aaindex1(data)
    test_data_ng, _ = ngram(data)
    test_data_ctd, _ = ctd(data)
    test_data_atc, _ = atc(data)
    test_data = np.concatenate(
        (test_data_aaindex1, test_data_ng, test_data_ctd, test_data_atc), axis=1#四个特征向量拼接到一起
    )
    indices = pd.read_csv("./train_data/indices_df.csv", header=None)
    indices = indices.iloc[:, 0]
    indices = np.array(indices)
    indices = indices.tolist()
    test_data = test_data[:, indices]

    train_data = pd.read_csv("./train_data/reduced_df.csv", header=None)
    train_data = np.array(train_data)
    train_target = pd.read_csv("./train_data/isAMP.csv", header=None)
    train_target = train_target.iloc[:, 0]
    train_target = np.array(train_target)

    return test_data, train_data, train_target
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
def add_bg_from_local(image_file):#背景
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local("DNA.jpg")
colindex = 0

# st.balloons()
# st.title("抗菌肽（AMP）预测")
# st.markdown("### 1.数据预处理")
centered_text = "<center><h1 >🧬 基于计算方法的抗菌肽预测模型</h1></center>" 
st.markdown(centered_text, unsafe_allow_html=True)
footer_html = """
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: black;
        text-align: center;
        padding: 10px;
    }
</style>
<div class="footer">
    <p>基于计算方法的抗菌肽预测模型 &copy;  学术机构：徐州医科大学</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

tab1, tab2, tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(["模型简介", "主要功能", "下载","使用说明","留言","关于我们","联系方式","链接"])
with tab2:
    col1, col2,= st.columns([1,2])
    with col1:
        st.subheader("📤上传肽序列数据或输入肽序列")
        test_file = st.file_uploader(
            "上传需要预测的肽数据(**CSV**格式，每行为一条肽序列)", type=["csv"]
        )
        if test_file is not None:
            colindex = 1
            testdata = pd.read_csv(test_file, header=0)
            test_data = testdata.iloc[:, 0] 
            test_data = np.array(test_data)
            test_data = test_data.tolist()
            st.write("肽序列数据预览，文件中共有" + str(len(test_data)) + "条肽序列")
            # 展示testdata第一列的数据预览
            st.write(testdata.iloc[:, 0])
        else:
            input_text = st.text_input("或者手动输入肽序列（多个肽序列请用逗号分隔）")
            colindex = 1
            if input_text:
                test_data = input_text.split(',')
                st.write("肽序列数据预览")
                st.write(test_data)
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
                    with st.spinner("预测中..."):# 显示旋转加载器
                        test_data_df, train_data, train_target = _feaprocess(test_data)  
                        if model == "随机森林 RF":  
                            rf_model = load_rf_model()  
                            pred = rf_model.predict_proba(test_data_df)[:, 1]  
                        elif model == "梯度提升 Gradient Boosting Classifier":  
                            gb_model = load_gb_model()  
                            pred = gb_model.predict_proba(test_data_df)[:, 1]  
                             
                        colindex = 2

            if colindex == 2:  
                st.subheader("📃 预测结果")  
                pred_proba = pd.DataFrame(pred, columns=["抗菌肽概率"])  
                result = pd.concat([pd.DataFrame(test_data, columns=["肽序列"]), pred_proba], axis=1)  
  
                # 应用样式（如果需要）  
                # 注意：这里不能直接通过style改变概率的颜色，但可以标记高概率的条目  
                high_prob_threshold = 0.7  # 假设我们认为概率大于0.7为高概率  
                 
  
                result.style.apply(lambda x: ['background-color: ' + x['高亮'] for x in x.to_dict().values()], axis=1)  
  
                # 汇总统计  
                high_prob_count = (result['抗菌肽概率'] > high_prob_threshold).sum()  
                st.write(f"高概率（>{high_prob_threshold}）是抗菌肽（AMP）的序列共{high_prob_count}条。")  
                #st.write(f"预测结果已显示，每行肽序列对应的'抗菌肽概率'列显示了其是抗菌肽的概率。")  
                st.write(result)  # 这里显示的是带有样式的 DataFrame  
  
                # 重新预测按钮  
                if st.button("重新预测"):  
                    st.experimental_rerun()

st.divider() 
with tab1:
    st.subheader("🔬 关于模型")
    st.markdown("基于计算方法的抗菌肽预测模型")
    st.markdown("---")  
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
    st.subheader('📥下载模型数据')
    data = {  
    'Name': ['Alice', 'Bob', 'Charlie'],  
    'Age': [25, 30, 35],  
    'City': ['New York', 'Los Angeles', 'Chicago']  
    }
    df = pd.DataFrame(data)  
    st.download_button(  
    label="下载训练集数据",  
    data=df.to_csv(index=False).encode('utf-8'),  
    file_name="train_data.csv",  
    mime="./train_data/reduced_df.csv"
    )
    st.download_button(  
    label="下载训练集数标签",  
    data=df.to_csv(index=False).encode('utf-8'),  
    file_name="train_target.csv",  
    mime="./train_data/isAMP.csv"
    )
    st.download_button(  
    label="下载特征选取所用数据",  
    data=df.to_csv(index=False).encode('utf-8'),  
    file_name="indices_df.csv",  
    mime="./train_data/indices_df.csv"
    )
with tab4:
    st.subheader('📚使用说明')
    st.markdown('1.上传数据，此处需要上传csv格式的数据。点击“Browse files”即可从文件夹中选择需要上传的蛋白质数据，数据需包含抗菌肽序列上传成功后可对数据进行预览。第一列为蛋白质的编号，第二列为蛋白质的序列展示，上下滚动可查看完整预览。或者手动输入蛋白质序列信息。')
    st.divider() 
    st.markdown('2.选择预测算法，可以选择我们微调好的随机森林或gbdt算法作为预测算法。每种算法选择下都有对该算法的简要说明，研究者可自行选用随机森林算法或gbdt算法，这两种算法都经过实验验证，性能高，预测能力强。')
    st.divider() 
    st.markdown('3.运行结果，经过运算的结果会显示在网页上，研究者可自主进行下载或查看操作。本应用将预测为抗菌肽的序列赋予标签“是”，将非抗菌肽的序列赋予标签“否”，研究者也可根据需要自行替换。表格上方会标出本次预测中抗菌肽和非抗菌肽的数量和占比。如对预测的结果不满意，可点击“重新预测”返回上一步重新运行模型。')
with tab5:
    MESSAGES_FILE = 'messages.json'  
  
# 加载留言函数  
    def load_messages():  
        if os.path.exists(MESSAGES_FILE):  
            with open(MESSAGES_FILE, 'r') as file:  
               messages = json.load(file)  
        else:  
            messages = []  
        return messages  
  
# 保存留言函数  
    def save_messages(messages):  
        with open(MESSAGES_FILE, 'w') as file:  
            json.dump(messages, file, indent=4)  
  
# 添加留言函数  
    def add_message(user, message, messages):  
        messages.append({"user": user, "message": message})  
        save_messages(messages)  # 提交后立即保存  
  
# 显示留言函数  
    def display_messages(messages):  
       for msg in messages:  
            st.write(f"{msg['user']}: {msg['message']}")  
  
# Streamlit应用主体  
    with st.container():  
        st.subheader("📝留言板")  
        messages = load_messages()  # 加载留言  
  
    # 用户输入留言  
        user = st.text_input("请输入你的名字：")  
        message = st.text_area("请输入你的留言：")  
        if st.button("提交留言"):  
            if user and message:  
                add_message(user, message, messages)  # 添加并保存留言  
                st.success("留言已提交！")  
            else:  
                st.warning("请填写完整信息！")  
  
    # 显示所有留言  
        st.subheader("留言列表：")  
        display_messages(messages)  # 显示加载的留言
with tab6:
    st.subheader("☎️ 联系我们")
    st.markdown("学术机构:徐州医科大学")




