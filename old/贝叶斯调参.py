# from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd

# # 训练集读取
train_data = pd.read_csv("aaindex1_train.csv", header=None)
train_data = np.array(train_data)
train_target = pd.read_csv("isAMP.csv", header=None)
train_target = train_target.iloc[:, 0]
train_target = np.array(train_target)

aaindex1_X_train, aaindex1_X_test, aaindex1_y_train, aaindex1_y_test = train_test_split(
    train_data, train_target, test_size=0.2, random_state=2
)

# 产生随机分类数据集，10个特征， 2个类别
# x, y = make_classification(n_samples=1000, n_features=5, n_classes=2)


# 步骤一：构造黑盒目标函数
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),  # float
            max_depth=int(max_depth),
            random_state=2,
        ),
        aaindex1_X_train,
        aaindex1_y_train,
        scoring="f1",
        cv=5,
    ).mean()
    return val


# 步骤二：确定取值空间
pbounds = {
    "n_estimators": (10, 250),  # 表示取值范围为10至250
    "min_samples_split": (2, 25),
    "max_features": (0.1, 0.999),
    "max_depth": (5, 15),
}

# 步骤三：构造贝叶斯优化器
optimizer = BayesianOptimization(
    f=rf_cv,  # 黑盒目标函数
    pbounds=pbounds,  # 取值空间
    verbose=2,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
    random_state=1,
)
optimizer.maximize(  # 运行
    init_points=5,  # 随机搜索的步数
    n_iter=25,  # 执行贝叶斯优化迭代次数
)
print(optimizer.res)  # 打印所有优化的结果
print(optimizer.max)  # 最好的结果与对应的参数
