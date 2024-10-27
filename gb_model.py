import pandas as pd  
import numpy as np  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
import joblib   
   
train_data = pd.read_csv("C:\\Users\\86151\\Desktop\\基于机器学习的抗菌肽预测模型\\代码和数据\\train_data\\reduced_df.csv", header=None)
train_data = np.array(train_data)
train_target = pd.read_csv("C:\\Users\\86151\\Desktop\\基于机器学习的抗菌肽预测模型\\代码和数据\\train_data\\isAMP.csv", header=None)
train_target = train_target.iloc[:, 0]
train_target = np.array(train_target)
 
gb = GradientBoostingClassifier(random_state=42)  
gb.fit(train_data, train_target)
joblib.dump(gb, 'gb_model.pkl')
