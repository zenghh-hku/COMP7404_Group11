import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# --- 步骤 1：读取数据并保留目标列 ---
keep_cols = ['Claim_Amount'] + [f'Cat{i}' for i in range(1, 12)]
train = pd.read_csv('/grp01/saas_lqqu/runxi/lightgbm/train_set.csv', nrows=10_000_000, usecols=keep_cols)
test = pd.read_csv('/grp01/saas_lqqu/runxi/lightgbm/train_set.csv', skiprows=range(1, 10_000_001), usecols=keep_cols)

# --- 步骤 2：处理 Claim_Amount 列 ---
def binarize_claim_amount(df):
    """将 Claim_Amount 非零值转为 1，零保持 0"""
    df['Claim_Amount'] = (df['Claim_Amount'] != 0).astype(int)
    return df

# 对训练集和测试集应用处理
train = binarize_claim_amount(train)
test = binarize_claim_amount(test)

# --- 步骤 3：分离目标变量 ---
y_train = train['Claim_Amount']
y_test = test['Claim_Amount']
X_train = train.drop(columns=['Claim_Amount'])
X_test = test.drop(columns=['Claim_Amount'])

# --- 步骤 4：One-Hot 编码离散特征 ---
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), [f'Cat{i}' for i in range(1, 12)])
    ],
    remainder='drop'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- 步骤 5：保存处理后的数据 ---
# 保存特征数据（稀疏矩阵格式）
pd.DataFrame.sparse.from_spmatrix(X_train_processed, columns=preprocessor.get_feature_names_out()) \
   .to_csv('/grp01/saas_lqqu/runxi/lightgbm/train_processed_features.csv', index=False)
pd.DataFrame.sparse.from_spmatrix(X_test_processed, columns=preprocessor.get_feature_names_out()) \
   .to_csv('/grp01/saas_lqqu/runxi/lightgbm/test_processed_features.csv', index=False)

# 保存目标变量（二值化后的 Claim_Amount）
y_train.to_csv('/grp01/saas_lqqu/runxi/lightgbm/train_labels.csv', index=False)
y_test.to_csv('/grp01/saas_lqqu/runxi/lightgbm/test_labels.csv', index=False)