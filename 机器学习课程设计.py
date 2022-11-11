import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier, plot_importance

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

test_CUST_ID = test_data["CUST_ID"]
train_data_target = train_data["bad_good"]
train_data.drop(["bad_good"], axis=1, inplace=True)

meaningless_columns = [
    "OPEN_ORG_NUM", "IDF_TYP_CD", "GENDER", "CUST_EUP_ACCT_FLAG", "CUST_AU_ACCT_FLAG",
    "CUST_DOLLER_FLAG", "CUST_INTERNATIONAL_GOLD_FLAG", "CUST_INTERNATIONAL_COMMON_FLAG",
    "CUST_INTERNATIONAL_SIL_FLAG", "CUST_INTERNATIONAL_DIAMOND_FLAG", "CUST_GOLD_COMMON_FLAG",
    "CUST_STAD_PLATINUM_FLAG", "CUST_LUXURY_PLATINUM_FLAG", "CUST_PLATINUM_FINANCIAL_FLAG", "CUST_DIAMOND_FLAG",
    "CUST_INFINIT_FLAG", "CUST_BUSINESS_FLAG"
]
# 删除没有意义的列
train_data.drop(meaningless_columns, axis=1, inplace=True)
test_data.drop(meaningless_columns, axis=1, inplace=True)

# 删除脏数据
train_data = train_data.drop_duplicates(keep="first")
train_data.dropna(inplace=True)

# Onehot 编码
num_col = train_data.select_dtypes(include=[np.number])
non_num_col = train_data.select_dtypes(exclude=[np.number])
onehotnum = pd.get_dummies(non_num_col)
train_data = pd.concat([num_col, onehotnum], axis=1)

num_col1 = test_data.select_dtypes(include=[np.number])
non_num_col1 = test_data.select_dtypes(exclude=[np.number])
onehotnum1 = pd.get_dummies(non_num_col1)
test_data = pd.concat([num_col1, onehotnum1], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    train_data, train_data_target, test_size=0.25
)

XGB = XGBClassifier(nthread=-1,  # 使用全部CPU进行并行运算
                    learning_rate=0.2,  # 学习率
                    n_estimators=100,  # 总共迭代100次，即决策树的个数
                    max_depth=5,  # 树的深度为5
                    gamma=0,  # 惩罚项系数
                    subsample=0.9,  # 训练每棵树时，使用的数据占全部训练集的比例
                    colsample_bytree=0.5)  # 训练每棵树时，使用的特征占全部特征的比例


# 对训练数据集进行评估
model = XGB.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("精确率（P）: ", precision_score(y_test, y_pred))
print("召回率（R）: ", recall_score(y_test, y_pred))
print("Macro-F1: ", f1_score(y_test, y_pred, average="macro"))

# 对测试数据进行预测
test_pred = model.predict(test_data)
test_pred = pd.DataFrame(test_pred, columns=["bad_good"])
sub = pd.concat([test_CUST_ID, test_pred], axis=1)
sub.to_csv('submission.csv')

